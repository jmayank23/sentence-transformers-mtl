# Task 2

Task 2 expands the single-task sentence embedding model (from Task 1) into a Multi-Task Learning (MTL) framework.

- Task A: Sentence classification (e.g., classifying sentences into 5 categories).
- Task B: Sentiment Analysis (e.g., classifying into 3 categories; pos, neg, neutral).

## Objectives
- Describe changes to model architecture from Task 1 to support multi-task learning

## Architectural Changes
1.	**Shared Sentence Encoder:**
The SentenceTransformer class (from Task 1) has been used, it outputs a fixed-dimensional embedding for each sentence using:
    - A pretrained BertModel.
    - A mean pooling layer over the sentence token embeddings.

2.	**Key Change - Addition of Task-Specific Heads:**
After obtaining the sentence embedding using SentenceTransformer, the embedding is passed into 2 heads:
    - classifier_taskA → A linear layer, outputs logits for Task A.
    - classifier_taskB → A linear later, outputs logits for Task B.

These task-specific heads operate on the shared sentence embeddings. Each head’s logits will then used to compute a task-specific loss (e.g., cross-entropy) during training.


