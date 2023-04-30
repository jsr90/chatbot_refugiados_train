# Chatbot Refugiados - Training

## Table of Contents

- [Script Summary](#script-summary)
- [Setup](#setup)
- [Installation](#installation)
- [Training](#training)
- [Specifications](#specifications)
- [SQuAD to custom](#squad_to_custom)


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
%cd /content/drive/Shareddrives/chatbot_refugiados
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

1. `--seed` (int): random seed
2. `--dir_path` (str): path to directory
3. `--path` (str): relative path to SQuAD file
4. `--wandb_entity` (str): W&B entity name
5. `--wandb_project` (str): W&B project name
6. `--wandb_log_model` (str): save your trained model checkpoint to wandb
7. `--wandb_watch` (str): turn off watch to log faster
8. `--train_batch_size` (str): per_device_train_batch_size
9. `--eval_batch_size` (str): per_device_eval_batch_size
10. `--evaluation_strategy` (int): 0:no, 1:steps, 2:epoch
11. `--num_train_epochs` (int): number of epochs during training
12. `--test_size` (float): size of test split. Default 0.2



Finally, to start the training process, run:

```
!python train.py
```

To customize the training process, simply pass the desired values for these arguments when running the script. If not provided, the script will use the default values specified above.

## Specifications

- The root directory must contain the following files and folders:

    - readme.md
    - train.py: Python file used for training the model.
    - utils.py: Python file with most of methods used.
    - data/data.json: SQuAD dataset

The `utils.py` module consists of several utility functions used in the Question Answering project. Here is a brief summary of the functions:

1. `transform_squad(path)`: This function reads a SQuAD file in JSON format and transforms it into a dictionary that can be easily used for training the model.

2. `get_train_test_val(path, test_size, seed)`: This function reads a SQuAD file and splits it into training and testing datasets. The path argument is the relative path to the SQuAD dataset, test_size is the size of the test partition (as a float between 0 and 1), and seed is a random seed to ensure reproducibility.

2. `set_vars(config)`: This function sets environment variables for WandB configuration. The config argument is an object that contains the necessary WandB configuration parameters (directory path, WandB entity, WandB project, WandB log model, and WandB watch).

3. `get_strategy(i)`: This function returns the evaluation strategy based on the given index. The i argument is the index of the evaluation strategy, and the possible values are "no", "steps", and "epoch".

4. `set_seed(seed)`: This function sets the random seed for reproducibility. The seed argument is the random seed value.

5. `preprocess_function(examples: dict, model_name)`: This function preprocesses the input data using the tokenizer. The examples argument is a dictionary containing the raw data, and the model_name argument is the name of the pre-trained model to be used for tokenization. The function returns a dictionary containing the preprocessed data.



