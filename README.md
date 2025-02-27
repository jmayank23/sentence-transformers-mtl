# Sentence Transformers & Multi-Task Learning

This repository contains an implementation of a Sentence Transformer and its extension to a Multi-Task Learning (MTL) framework, built using `PyTorch` and the Hugging Face `transformers` library. It demonstrates the creation of sentence embeddings, multi-task learning with shared representations, and training considerations (freezing layers and transfer learning) for such models.


## Tasks

*   **Task 1:** Implements a basic Sentence Transformer using a pre-trained BERT model and mean pooling.  (`src/task1`)
*   **Task 2:** Extends the Sentence Transformer to handle multi-task learning (sentence classification and sentiment analysis, as examples). (`src/task2`)
*   **Task 3:** Discusses training considerations, including freezing strategies and transfer learning. (`src/task3`)
*   **Task 4:** Implements a training loop for the multi-task model. (`src/task4`)

### READMEs

Each task directory contains a `taskX-readme.md` file (where X is the task number) that provides detailed explanations, design choices, and discussions related to that specific task.

## Setup and Installation

### Option 1: Pulling from Docker Hub

Pull the Docker image and run it:

```bash
docker run -it jmayank/sentence-transformer-mtl
```

### Option 2: Using a Virtual Environment

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/jmayank23/sentence-transformers-mtl.git
    cd sentence-transformers-mtl
    ```

2.  **Create a virtual environment:**

    Using `venv`:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    This file contains a list of necessary Python packages
        
## Running the Code

Run the following commands from the project directory. The `-m` flag has been used to make loading of modules easier in each task's code. 

_Note: `__init__.py` files were created inside each task directory to make importing models easier._

*   **Task 1 (Sentence Transformer):**

    ```bash
    python -m src.task1.sentence_transformer
    ```
    This will download the `bert-base-uncased` model (if not already cached) and print the embeddings for a few example sentences.

*   **Task 2 (Multi-Task Model):**

    ```bash
    python -m src.task2.sentence_transformer_mtl
    ```
    This will instantiate the multi-task model and print the shapes and example logits for two tasks.

*   **Task 4 (Training Loop):**

    ```bash
    python -m src.task4.train_mtl
    ```
     This will train the multi-task model on a dummy dataset for a few epochs and print the loss and accuracy for each task.