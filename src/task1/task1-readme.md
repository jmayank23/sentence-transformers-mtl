# Task 1

This directory contains an implementation of a **Sentence Transformer** to produce embeddings for sentences. It uses `PyTorch` and the `transformers` library from HuggingFace.

## Objectives 
- Encode input sentences into fixed-length embeddings.  
- Test the implementation with a few sample sentences.
- Describe design choices

## Key Design Choices

1. **Pooling Mechanism**: Mean Pooling
    - **Rationale**: Mean pooling is used because it aggregates information from *all* tokens in the sentence rather than relying on a single `[CLS]` position. This often provides a more holistic representation, capturing nuances from every token. 

    - Why not just `[CLS]`?
        - The `[CLS]` token's effectiveness depends on the pre-training objective.  
        - The `bert-base-uncased` model used `[CLS]` for Masked Language Modeling (MLM) and Next Sentence Prediction (NSP), which is not guaranteed to be optimal for *all* downstream tasks. 
        - Additionally, state-of-the-art models like [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) use mean pooling, showing its practical effectiveness.

   _Note: In practice, both `[CLS]` token embedding and mean pooling are commonly used and which one to use can be decided based on whichever yields better performance for the particular task at hand._

## Model Architecture

- **Transformer Backbone**: Uses the pretrained `bert-base-uncased` model
  
- **MeanPooling**: A class that performs mean pooling on token embeddings, weighted by the attention mask. This ensures that only valid tokens contribute to the final sentence embedding.