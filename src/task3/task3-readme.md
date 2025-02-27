# Task 3

There is no code for this task, the training and transfer learning scenarios are discussed here. 

---

## Objectives
1. **Understand the implications of freezing** different model components (the entire network, only the transformer backbone, or one of the task-specific heads).
2. **Explore how transfer learning** can be beneficial for multi-task setups, including:
   - Which pretrained model to use,
   - Which layers to freeze or unfreeze,
   - The rationale behind these decisions.
3. Brief write up summarizing key decisions and insights

---

## Freezing Scenarios

### 1. Freezing the Entire Network
- **Use Case**: Deploy a pretrained model for inference without fine-tuning (e.g., zero-shot evaluation or rapid prototyping).  
- **Implication**: No parameters are updated; the model retains its original pretrained behavior.  
- **Advantage**: 
  - Relatively lower compute requirement
  - Preserves all pretrained capabilities (no catastrophic forgetting).
- **Disadvantage**: The model may not adapt to new tasks/domains, so performance on these new tasks may be suboptimal.

### 2. Freezing Only the Transformer Backbone
- **Use Case**: Adapt a general-purpose model to new tasks with limited data (e.g., sentiment analysis on product reviews).  
- **Implication**: The parameters of the sentence encoder (the part that produces an embedding for the sentence) remain fixed and only the parameters of the task-specific heads are updated.  
- **Advantage**:  
  - Faster training (fewer parameters updated). 
  - Preserves the capabilities of the backbone to capture meaning from underlying input data and allows adaptation of the model to the specific task(s).  
- **Disadvantage**: If the backbone’s representations are not well-aligned with the new tasks, performance gains may be limited.
  - For example, consider `bert-base-uncased` being used to analyse (for example, sentiment analysis) transcripts of earnings calls. The frozen backbone may produce embeddings that fail to capture domain-specific nuances. For example, "bullish" in english is associated with bulls or aggression, but in finance it explicitly signals a positive sentiment.

### 3. Freezing Only One of the Task-Specific Heads
- **Use Case**: Preserve performance on Task A (e.g., sentiment analysis) while training Task B (e.g., named entity recognition) and/or the backbone.
- **Implications**: The transformer backbone and the other task head remain trainable, while one head is kept fixed.  
    - Backbone frozen: 
      - _Safe_: Task A’s performance is preserved (representations stay fixed).
      - Task B’s head adapts to the frozen backbone’s features.
    - Backbone unfrozen:
      - _Risky_: Task A's head may fail due to changes in the underlying representation (drift).
- **Advantages**: 
  - Might enable incremental updates (e.g., add Task B without disrupting Task A).
- **Disadvantages**: 
  - It is only effective if backbone is frozen; unfrozen backbones risk task misalignment.
---

## Transfer Learning Considerations

In general, if we want to train a model for a specific domain, for example, finance, it can be beneficial to start with a generally capable model and then fine-tune on domain-specific/ proprietary company data. More details on this example below.

### 1. Choice of a Pretrained Model
- **General Domain**: If the tasks are broad (e.g., general classification or sentiment analysis), popular models like `bert-base-uncased` are good starting points.  
- **Domain-Specific**: For specialized fields such as finance or biomedical text, domain-specific variants (e.g., FinBERT, BioBERT) can be good starting points as they already have some understanding of domain-related terminology and context.

### 2. Layers to Freeze/Unfreeze
The choice of layers to freeze/unfreeze depends on the complexity of the task and the amount of data available for fine-tuning.
- **Partial Freezing**:  
  - The lower layers are frozen to preserve the general "meaning" that the pretrained models capture.
  - Upper layers are unfrozen to fine-tune for the specific downstream tasks.  
- **Progressive Unfreezing**: We can start by unfreezing only the topmost layer, fine-tuning the model and then gradually unfreeze more layers if we continue seeing improvement.  

### 3. Rationale
- **Data Availability**:  
  - Small datasets: Freeze most/ all layers to avoid overfitting.
  - Large datasets: Unfreeze more layers for better adaptation.  
- **Domain Similarity**:  
  - High similarity: Fewer layers unfronzen (For example, general -> finance)
  - Low similarity: More layers unfrozen (For example, general -> medical domain tasks)
- **Resource Constraints**:  
  - Freezing layers significantly reduces training time and memory usage.

### 4. Example Scenario
Suppose we are trying to adapt a general-purpose BERT to a finance domain:
1. **Start** with `bert-base-uncased`.  
2. **Freeze** the majority of its layers, and only unfreeze the top few layers and any newly added classification heads.  
3. **Fine-tune** on a set of finance-related texts to adjust the model to finance terminology.  
4. **Evaluate**: If performance is still lacking, progressively unfreeze additional layers for a deeper adaptation. Note, if the model is overfitting to our data, then we may have to collect more data or reduce unfrozen layers.
---

## Write up on Key Decisions and Insights

### Key Decisions
- Layer Freezing Strategies
  - Recommended selective freezing based on available compute, data availability and task similarity between pre-training and fine-tuning task. 
    - Limited compute availability / Less data / High task similarity -> Freeze more layers
    - Higher compute availability / More data / Low task similarity-> Freeze lesser layers
  - Generally, freezing the backbone is a good starting point.
  - Progressive unfreezing of layers for domain adaptation can be a good strategy

- Transfer Learning Approaches
  - Recommended using domain-speciifc pretrained models if avaialble and then fine-tune on new task/ data.

### Key Insights
- The choice of freezing strategy significantly impacts both performance and computational requirements, with backbone freezing offering the best efficiency-adaptation balance for most scenarios.
- Transfer learning effectiveness depends on domain similarity between pretraining and target tasks - the closer the domains, the more layers can remain frozen.
- Component (backbone and task specific heads) interdependence means freezing one component doesn't isolate it from performance changes, as shared representations can continue to evolve through other components.
