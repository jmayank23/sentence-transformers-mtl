import os
from torch import nn
from transformers import BertTokenizerFast
from src.task1.sentence_transformer import SentenceTransformer

class SentenceTransformerMTL(nn.Module):
    """
    A multi-task learning model that builds on top of the SentenceTransformer to handle
    multiple NLP tasks simultaneously.
    
    Chosen tasks:
        - Task A: Sentence Classification (e.g., 5 classes)
        - Task B: Sentiment Analysis (e.g., 3 classes; pos/neg/neutral)
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        cache_dir: str = None,
        num_labels_taskA: int = 5,
        num_labels_taskB: int = 3
    ):
        """
        Initializes the MultiTaskSentenceTransformer.

        Args:
            model_name:      The name of the pre-trained model to use for the backbone.
            cache_dir:       Directory in which to store and look for model weights.
            num_labels_taskA: Number of labels for Task A classification.
            num_labels_taskB: Number of labels (or output units) for Task B classification.
        """
        super().__init__()
        self.sentence_transformer = SentenceTransformer(model_name, cache_dir=cache_dir)
        
        # Task-specific heads:
        self.classifier_taskA = nn.Linear(768, num_labels_taskA) 
        self.classifier_taskB = nn.Linear(768, num_labels_taskB)

    def forward(self, inputs: dict):
        """
        Forward pass for the multi-task model.

        Args:
            inputs: A dictionary containing 'input_ids', 'attention_mask', etc.

        Returns:
            A tuple of logits: (taskA_logits, taskB_logits)
        """
        # Get the shared sentence embeddings
        embeddings = self.sentence_transformer(inputs)

        # Each head produces its own logits
        logits_taskA = self.classifier_taskA(embeddings)
        logits_taskB = self.classifier_taskB(embeddings)

        return logits_taskA, logits_taskB
    

# ------------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------------

if __name__=='__main__':
    cache_dir = os.path.join(os.getcwd(), "model_weights")
    model_name = "bert-base-uncased"
    num_labels_taskA = 5
    num_labels_taskB = 3

    tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)

    # Instantiate the multi-task model
    multi_task_model = SentenceTransformerMTL(
        model_name=model_name,
        cache_dir=cache_dir,
        num_labels_taskA=num_labels_taskA,
        num_labels_taskB=num_labels_taskB
    )

    sentences = ["Hello, world!", "Programming is very powerful.", "This is a test sentence."]
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    taskA_logits, taskB_logits = multi_task_model(inputs)

    print("Task A logits shape:", taskA_logits.shape)  # (batch_size, num_labels_taskA)
    print("Task B logits shape:", taskB_logits.shape)  # (batch_size, num_labels_taskB)
    print("\nTask A logits:", taskA_logits)
    print("Task B logits:", taskB_logits)