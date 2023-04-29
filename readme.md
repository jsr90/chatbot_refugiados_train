# Chatbot Refugiados - Training

## Table of Contents

- [Script Summary](#script-summary)
- [Setup](#setup)
- [Installation](#installation)
- [Training](#training)
- [Specifications](#specifications)


## Script Summary

This script trains a question-answering model for the "Chatbot Refugiados" project using the provided configuration. The following is a brief summary of the script's main steps:

1. Import required libraries and modules, such as Weights & Biases (W&B) for logging and visualization, the Hugging Face Transformers library for the model and tokenizer, and the Fastai library for the configuration object.
2. Set up default configuration values and define functions to parse command-line arguments, set environment variables, and set the random seed for reproducibility.
3. Define functions for data preprocessing, metric computation, dataset preparation, and model training.
4. Download the pre-trained model and set up the training arguments using the Hugging Face Transformers library.
5. Load and preprocess the dataset.
6. Train the model using the Hugging Face Trainer class.
7. Finish the Weights & Biases run.

## Setup

First, you need to mount your Google Drive and navigate to the folder containing the project (make sure you specify your own path):

```python
from google.colab import drive
drive.mount("/content/drive/", force_remount=True)
%cd /content/drive/Shareddrives/chatbot_refugiados/Chatbot_refugiados
```

## Installation

Next, install the required dependencies by running the following command:

```
!pip install -r requirements.txt
```

## Training

Before you start training, you can see the available command-line options and their descriptions by running:
```
!python train.py --help
```

The script accepts the following command-line arguments that allow you to customize the training process:

1. `--seed`: (int) Random seed for reproducibility. Default: 42.
2. `--dir_path`: (str) Path to the directory containing the project files. Default: "/content/drive/Shareddrives/chatbot_refugiados/Chatbot_refugiados/".
3. `--wandb_entity`: (str) Weights & Biases entity name. Default: "jesus-saturdays/saturdays".
4. `--wandb_project`: (str) Weights & Biases project name. Default: "chatbot_refugees".
5. `--wandb_log_model`: (str) Save your trained model checkpoint to Weights & Biases. Default: "true".
6. `--wandb_watch`: (str) Turn off watch to log faster in Weights & Biases. Default: "true".
7. `--train_batch_size`: (str) Per-device train batch size. Default: 16.
8. `--eval_batch_size`: (str) Per-device evaluation batch size. Default: 16.
9. `--evaluation_strategy`: (int) Evaluation strategy, with the following options: 0 for "no", 1 for "steps", and 2 for "epoch". Default: 2.
10. `--num_train_epochs`: (int) Number of epochs during training. Default: 25.

Finally, to start the training process, run:

```
!python train.py
```

To customize the training process, simply pass the desired values for these arguments when running the script. If not provided, the script will use the default values specified above.

## Specifications

- The root directory must contain the following files and folders:

    - readme.md
    - train.ipynb: Jupyter Notebook file used for execution in Google Colab.
    - train.py: Python file used for training the model.
    - data/: Folder containing:
        - train.json
        - eval.json files.

- The format of the train.json and eval.json files must be as follows:

```json
{
    "data": [
        {
            "answers": {
                "answer_start": int,
                "text": [str]
            },
            "context": str,
            "id": int,
            "question": str
        }        
    ]
}
 ```