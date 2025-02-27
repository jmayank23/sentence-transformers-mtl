import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.task2.sentence_transformer_mtl import SentenceTransformerMTL

###############################################################################
# 1. Hypothetical Data: Multi-Task Dataset
###############################################################################

class DummyMTLDataset(Dataset):
    """
    A dummy dataset simulating multi-task data for demonstration:
    - Task A: 5-class classification
    - Task B: 3-class classification
    Each sample is composed of:
      - input_ids
      - attention_mask
      - labelA
      - labelB
    """
    def __init__(self, num_samples=50, seq_len=12, num_labels_a=5, num_labels_b=3):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_labels_a = num_labels_a
        self.num_labels_b = num_labels_b

        # Fake token IDs in [0, 30522) for a model like BERT
        self.input_ids = torch.randint(0, 30522, (num_samples, seq_len))
        self.attention_mask = torch.ones((num_samples, seq_len))

        # Random integer labels for each task
        self.labelsA = torch.randint(0, num_labels_a, (num_samples,))
        self.labelsB = torch.randint(0, num_labels_b, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labelA": self.labelsA[idx],
            "labelB": self.labelsB[idx]
        }


###############################################################################
# 2. Training Loop
###############################################################################

def train_mtl(model, dataloader, epochs=3, lr=1e-4):
    """
    Trains the multi-task model on two classification tasks.

    Args:
        model: The SentenceTransformerMTL instance from Task 2.
        dataloader: A DataLoader that yields (inputs, labelsA, labelsB).
        epochs: Number of training epochs.
        lr: Learning rate.
    """
    # We use CrossEntropyLoss for both tasks
    criterionA = nn.CrossEntropyLoss()
    criterionB = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        total_correct_a = 0
        total_correct_b = 0

        for batch in dataloader:
            batch_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"]
            }
            labelsA = batch["labelA"]
            labelsB = batch["labelB"]
            
            logitsA, logitsB = model(batch_inputs) # FORWARD PASS
            
            lossA = criterionA(logitsA, labelsA)
            lossB = criterionB(logitsB, labelsB)
            
            # Combine losses (simple sum)
            loss = lossA + lossB
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labelsA.size(0)
            total_loss += loss.item() * batch_size # loss.item() returns average loss per sample
            total_samples += batch_size

            # Calculate accuracy
            preds_a = torch.argmax(logitsA, dim=1)
            correct_a = (preds_a == labelsA).sum().item()
            total_correct_a += correct_a

            preds_b = torch.argmax(logitsB, dim=1)
            correct_b = (preds_b == labelsB).sum().item()
            total_correct_b += correct_b


        ###############################################################################
        # 3. Metrics
        ###############################################################################
        avg_loss = total_loss / total_samples

        epoch_acc_a = total_correct_a / total_samples
        epoch_acc_b = total_correct_b / total_samples
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f} | "
            f"Acc-A: {epoch_acc_a:.2f} | Acc-B: {epoch_acc_b:.2f}")

        
###############################################################################
# Main Execution
###############################################################################

# Initialize the model
mtl_model = SentenceTransformerMTL(
    model_name="bert-base-uncased",
    cache_dir=os.path.join(os.getcwd(), "model_weights"),
    num_labels_taskA=5,
    num_labels_taskB=3
)

# Create a dummy dataset and dataloader
dataset = DummyMTLDataset(
    num_samples=10, seq_len=12, # max seq len for bert-base-uncased is 512
    num_labels_a=5, num_labels_b=3
)
dataloader = DataLoader(
    dataset, batch_size=8, 
    shuffle=True, 
)

# Train the model
train_mtl(
    model=mtl_model,
    dataloader=dataloader,
    epochs=2,
    lr=1e-4
)