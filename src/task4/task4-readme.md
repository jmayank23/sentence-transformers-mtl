# Task 4

This task extends the SentenceTransformer MTL model from Task 2 by providing a **multi-task training loop** for simultaneously training on two tasks (e.g., Task A with 5 classes, Task B with 3 classes) using a shared transformer encoder.

---

## Objectives
- Define training loop for SentenceTransformerMTL (model from Task 2)
- Elabroate on:
   - Handling of hypothetical data
   - Forward Pass
   - Metrics
- Brief write up summarizing key decisions and insights

---

## Key Components

1. **Handling Hypothetical Data:** `DummyMTLDataset`  
   - Yields random `input_ids`, `attention_mask`, and labels (`labelA`, `labelB`).  
      - Random token IDs (0-30522 range for BERT compatibility)
      - Synthetic labels for 2 classification tasks

2. **Forward Pass**  
   - For each batch of inputs received from the DataLoader, the model outputs (`logitsA, logitsB`).  
   - Task-specific losses (`lossA`, `lossB`) are computed and then combined to get `total_loss`.

3. **Metrics**
   - **Task-Specific Accuracy**: Tracks accuracy for both tasks separately
   - **Combined Loss**: Sum of both task losses

4. **Assumptions and Decisions Made**  
   - **Loss Summation**: CrossEntropyLoss was used for both tasks and combined to get a single loss function, however, it is also possible to define different loss functions for both tasks.
      - Could also use a weighted loss 
   - **Metrics**: Accuracy may not be the best metric to track if there is class imbalance. 
      - F1 Score would be better as it balances precision and recall, preventing models from achieving high accuracy by simply predicting the majority class
      - Unlike accuracy, F1 score accounts for both false positives and false negatives, making it more robust for uneven class distributions
   - Device management: Current code assumed CPU-only. If GPU is available, perform `.to(device)` to move model and data to the GPU for training.

---

## Multi-Task Training Dynamics

### Task Interference and Balance
Training multiple tasks simultaneously introduces several challenges which would have to be addressed in a scenario with real data, below is a discussion on some of these challenges.

1. **Task Competition and Interference**
   - **Challenge**: Tasks may compete for model capacity or even work against each other
     - Task A (5-class classification) might require different features than Task B (sentiment analysis)
     - One task could dominate training due to complexity or loss magnitude differences
   - **Implementation Choice**: The simple summation of losses (`loss = lossA + lossB`) assumes:
     - Both tasks are equally important
     - Loss scales are comparable between tasks
   - **Mitigation Strategies**:
     - Weighted summation: `loss = alpha * lossA + (1-alpha) * lossB`
     - Dynamic weighting based on task difficulty or validation performance
     - Task-specific gradient clipping to prevent one task from dominating updates

2. **Convergence Rate Differences**
   - **Challenge**: Tasks typically learn at different speeds
     - Task B (for example, sentiment analysis) may converge faster than the more complex Task A (for example, a 5-class classification)
     - Early stages of training might favor easier tasks
   - **Implementation Approach**:
     - A single optimizer and learning rate has been used for both tasks
     - Metrics for both tasks are tracked to detect imbalanced learning
   - **Mitigation Strategies**:
     - Task-specific learning rate scheduling
     - Curriculum learning (gradually increasing difficult tasks' importance). Can be done by
         - Implementing a weighted loss 
         - Sampling some examples more frequently

3. **Training Stability**
   - **Challenge**: Multi-task models can be more unstable during training
     - Conflicting gradients may lead to oscillations
     - Early overfitting on one task can harm shared representations
   - **Implementation Considerations**:
     - Adam optimizer was chosen because it allows adaptive learning rates across parameters
     - Training loop tracks metrics for both tasks to monitor stability

4. **Gradient Conflicts**
   - **Challenge**: Updates that help one task might harm another
   - **Mitigation Strategies**:
     - Normalize gradient magnitudes
     - Explore more strategies thorough a literature search 

These considerations inform the implementation choices while highlighting potential extensions for more sophisticated multi-task training approaches.

---

## Write up on Key Decisions and Insights
### Key Decisions
- Strategy related to metrics:
   - Implemented simple loss summation as baseline approach
   - Acknowledged limitations and proposed weighted alternatives
   - Designed training loop to track task-specific metrics separately

- Optimizer Selection:
   - Chose Adam optimizer for adaptive learning rates
   - Used conservative learning rate (1e-4) to ensure some training stability
   - Suggested task-specific learning rates as a potential improvement
- Evaluation Approach:
   - Performed separate accuracy tracking for each task
   - Used combined loss as overall training signal and acknowledged potential challenges from gradient interference in a multi task training setting
   - Recognized limitations of accuracy for imbalanced datasets

### Key Insights

- Multi-task learning introduces unique training dynamics where tasks compete for model capacity, making careful gradient management and optimization essential.
   - Monitoring training and validation accuracy for both tasks over time is crucial.
- Task interference can be both beneficial (positive transfer) and detrimental (negative transfer), necessitating careful monitoring and mitigation.
- Different convergence rates between tasks call for specialized handling through techniques like gradient normalization, task weighting, or curriculum learning.
- The current training loop is a basic starting point, and more sophisticated strategies should be adopted based on training outcomes from real data.