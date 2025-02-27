import os
import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast

class SentenceTransformer(nn.Module):
    """
    Sentence Transformer class that encodes input sentences into fixed-length embeddings
    using a BERT backbone and a mean pooling strategy.
    """
    def __init__(self, model_name: str = "bert-base-uncased", cache_dir: str = None):
        """
        Initializes the SentenceTransformer.

        Args:
            model_name: The name of the pre-trained model to use.
            cache_dir:  Directory in which to store and look for model weights.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.pooler = MeanPooling()

    def forward(self, inputs: dict):
        """
        Encodes inputs into sentence embeddings.

        Args:
            inputs: A dictionary containing 'input_ids', 'attention_mask', and 'token_type_ids'.

        Returns:
            A tensor of shape (batch_size, embedding_dimension) containing sentence embeddings.
        """
        outputs = self.bert(**inputs)
        token_embeddings = outputs.last_hidden_state # (batch_size, sequence_length, hidden_size)

        sentence_embedding = self.pooler(token_embeddings, inputs["attention_mask"]) # (batch_size, hidden_size)
        # sentence_embedding = token_embeddings[:, 0, :]  # CLS token embedding, kept here only for completeness.

        return sentence_embedding


class MeanPooling(nn.Module):
    """
    Performs mean pooling on token embeddings, weighted by the attention mask.
    """
    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        """
        Computes mean-pooled embeddings for each sentence.

        Args:
            token_embeddings: Tensor of token embeddings from the transformer.
                             Shape: (batch_size, sequence_length, embedding_dimension).
            attention_mask:   Attention mask indicating which tokens are real vs. padding.
                             Shape: (batch_size, sequence_length).

        Returns:
            Tensor of sentence embeddings. Shape: (batch_size, embedding_dimension).
        """
        # Expand the attention mask to match the embedding dimensions:
        # (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dimension).
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Weighted sum of token embeddings, zeroing out padding tokens.
        # (batch_size, sequence_length, embedding_dimension) * (batch_size, sequence_length, embedding_dimension) -> (batch_size, embedding_dimension)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # Sum of the expanded mask to count real tokens.
        sum_mask = input_mask_expanded.sum(dim=1) # (batch_size, 1)

        # Clamp the sum to avoid division by zero.
        sum_mask = torch.clamp(sum_mask, min=1e-9) # (batch_size, 1)

        # Compute the mean by dividing summed embeddings by the number of real tokens.
        # (batch_size, embedding_dimension) / (batch_size, 1) -> (batch_size, embedding_dimension)
        return sum_embeddings / sum_mask


# ------------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------------

if __name__=='__main__':
    cache_dir = os.path.join(os.getcwd(), "model_weights")
    model_name = "bert-base-uncased"

    tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
    model = SentenceTransformer(model_name=model_name, cache_dir=cache_dir)

    sentences = ["Hello, world!", "This is a test."]
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    embeddings = model(inputs)

    for s, emb in zip(sentences, embeddings):
        print("-" * 100)
        print(f"Sentence: {s}")
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding: {emb}")

    print("-" * 100)
