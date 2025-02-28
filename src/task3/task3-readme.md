# Task 3

There is no code for this task, the training and transfer learning scenarios are discussed here. 

---

## Objectives
1. **Understand the implications of freezing** different model components
  - The entire network, on
  - Only the transformer backbone
  - Only one of the task-specific heads
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
- **Disadvantage**: The model may not adapt to new tasks/domains, so performance on new tasks may be suboptimal.

### 2. Freezing Only the Transformer Backbone
- **Use Case**: Adapt a general-purpose model to new tasks with limited data (e.g., sentiment analysis on product reviews).  
- **Implication**: The parameters of the sentence encoder (the part that produces an embedding for the sentence) remain fixed and only the parameters of the task-specific heads are updated.  
- **Advantage**:  
  - Faster training (fewer parameters updated). 
  - Preserves the capabilities of the backbone to capture meaning from underlying input data and allows adaptation of the model to the specific task(s).  
- **Disadvantage**: 
  -	Limited Adaptability: If the backbone’s learned representations do not capture the nuances of the new task (e.g., specialized medical terminology), then freezing it may hinder performance improvements.
  - Potential Misalignment: The task-specific heads might struggle to adapt to features that were not emphasized during pretraining. (e.g., a term like “bullish” may be interpreted differently in a finance context versus general usage).

### 3. Freezing Only One of the Task-Specific Heads
- **Use Case**: Preserve performance on Task A (e.g., sentiment analysis) while training Task B (e.g., named entity recognition) and/or the backbone.
- **Implications**: The transformer backbone and the other task head remain trainable, while one head is kept fixed.  
    - Backbone frozen: _Safe_
      - Task A’s performance is preserved (representations stay fixed).
      - Task B’s head adapts to the frozen backbone’s features.
    - Backbone unfrozen: _Risky_
      - Task A's head may fail due to changes in the underlying sentence representation.
- **Advantages**: 
  - Might enable incremental updates (e.g., add Task B without disrupting Task A).
- **Disadvantages**: 
  - It is only effective if backbone is frozen; unfrozen backbones risk task misalignment.
---

## Transfer Learning Considerations

If we want to train a model for a specific domain, for example, finance, it can be beneficial to start with a generally capable model (i.e., a pre-trained model) and then fine-tune on domain-specific/ proprietary data. More details on this example below.

### 1. Choice of a Pretrained Model
- **General Domain**: If the tasks are broad (e.g., general classification or sentiment analysis), popular models like `bert-base-uncased` are good starting points.  
- **Domain-Specific**: For specialized fields such as finance or biomedical text, domain-specific variants (e.g., FinBERT, BioBERT) can be good starting points as they already have some understanding of domain-related terminology and context.

### 2. Layers to Freeze/Unfreeze
The choice of layers to freeze/unfreeze depends on the complexity of the task, amount of data available for fine-tuning and the compute resources that are available. 

- Lower layers: Capture basic syntactic structures and local dependencies
- Intermediate Layers: Integrate information over longer distances within the sentence, capturing complex patterns and dependencies.
- Higher layers: Capture high-level semantic information

A good strategy to decide layers to freeze/ unfreeze is to start with freezing only the top-most layer, training on data, monitor results and the progressively decide to unfreeze more layers.

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
  - Recommended using domain-specifc pretrained models if available and then fine-tune on new task/ data.

### Key Insights
- The choice of freezing strategy significantly impacts performance and computational efficiency. For many scenarios, freezing the backbone offers an optimal balance between efficiency and adaptability.
-	The effectiveness of transfer learning largely depends on the similarity between the pretraining domain and the target domain; closer domains allow for more layers to remain frozen.
-	Interdependencies between the backbone and task-specific heads mean that changes in one component can affect overall performance. Even when freezing one component, shared representations may evolve if other parts of the network are updated.