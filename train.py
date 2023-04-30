import wandb
import os
import numpy as np
import argparse
import random, torch
from fastai.vision.all import SimpleNamespace
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

evaluation_strategies = ["no", "steps", "epoch"]

# Define default configuration for the experiment
config = SimpleNamespace(
    framework="fastai",
    dir_path="./",
    wandb_entity="jesus-saturdays/saturdays",
    wandb_project="chatbot_refugees",
    seed=42,
    wandb_log_model="true",
    wandb_watch="true",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy = 2,
    model_name = "timpal0l/mdeberta-v3-base-squad2",
    num_train_epochs=25
)

def parse_args() -> None:
    """
    Override default arguments with the ones provided by the user.
    """
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    argparser.add_argument('--dir_path', type=str, default=config.dir_path, help='path to directory')
    argparser.add_argument('--wandb_entity', type=str, default=config.wandb_entity, help='W&B entity name')
    argparser.add_argument('--wandb_project', type=str, default=config.wandb_project, help='W&B project name')
    argparser.add_argument('--wandb_log_model', type=str, default=config.wandb_log_model, help='save your trained model checkpoint to wandb')
    argparser.add_argument('--wandb_watch', type=str, default=config.wandb_watch, help='turn off watch to log faster')
    argparser.add_argument('--train_batch_size', type=str, default=config.per_device_train_batch_size, help='per_device_train_batch_size')
    argparser.add_argument('--eval_batch_size', type=str, default=config.per_device_eval_batch_size, help='per_device_eval_batch_size')
    argparser.add_argument('--evaluation_strategy', type=int, default=config.evaluation_strategy, help='0:no, 1:steps, 2:epoch')
    argparser.add_argument('--num_train_epochs', type=int, default=config.num_train_epochs, help='number of epochs during training')
    args = argparser.parse_args()
    vars(config).update(vars(args))
    return

def set_vars() -> None:
    """
    Set environment variables for wandb configuration.
    """
    os.environ["WANDB_DIR"]=config.dir_path
    os.environ["WANDB_ENTITY"]=config.wandb_entity
    os.environ["WANDB_PROJECT"]=config.wandb_project
    os.environ["WANDB_LOG_MODEL"]=config.wandb_log_model
    os.environ["WANDB_WATCH"]=config.wandb_watch

def get_strategy(i: int) -> str:
    """
    Get evaluation strategy based on the given index.
    Args:
        i (int): Index of the evaluation strategy.
    Returns:
        str: The evaluation strategy (either "no", "steps", or "epoch").
    """
    try:
        evaluation_strategy = evaluation_strategies[i]
    except:
        print("Incorrect evaluation strategy index. Set to 2.")
        evaluation_strategy = evaluation_strategy[2]
    return evaluation_strategy

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The random seed value.
    """

    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch's random number generator for CPU and GPU
    torch.manual_seed(seed)

    # Set the seed for PyTorch's random number generator for all devices (both CPU and GPU)
    torch.cuda.manual_seed_all(seed)

    # Make the selection of convolution algorithms deterministic in PyTorch
    torch.backends.cudnn.deterministic = True

    # Disable the benchmark mode in PyTorch, which allows the library to select the best algorithm
    # for the hardware, but may result in non-deterministic results
    torch.backends.cudnn.benchmark = False


def preprocess_function(examples: dict) -> dict:
    """
    Preprocess the input data using the tokenizer.
    Args:
        examples (dict): Dictionary containing raw data.
    Returns:
        dict: Dictionary containing the preprocessed data.
    """

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
      questions,
      examples["context"],
      max_length=384,
      truncation="only_second",
      return_offsets_mapping=True,
      padding="max_length",
  )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer['text'][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def compute_metrics(eval_pred: tuple) -> dict:
    """
    Compute accuracy metric for evaluation.
    Args:
        eval_pred (tuple): Tuple containing logits and labels.
    Returns:
        dict: Dictionary containing the accuracy metric.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}

def prepare_data() -> dict:
    """
    Load and preprocess the dataset.
    Returns:
        dict: Dictionary containing the tokenized dataset.
    """
    squad_train_data_path = config.dir_path+'data/train.json'
    squad_eval_data_path = config.dir_path+'data/eval.json'
    dataset = load_dataset('json', data_files={'train': squad_train_data_path, 'eval': squad_eval_data_path}, field='data')
    tokenized_squad = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
    return tokenized_squad

def train(config: SimpleNamespace) -> None:
    """
    Train the model using the provided configuration.
    Args:
        config (SimpleNamespace): Configuration object containing experiment settings.
    """
    # download the model
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_name)
    # pass "wandb" to the 'report_to' parameter to turn on wandb logging
    training_args = TrainingArguments(
        output_dir=config.dir_path+'models',
        num_train_epochs=config.num_train_epochs,
        report_to="wandb",
        logging_steps=10, 
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        evaluation_strategy=get_strategy(config.evaluation_strategy),
        eval_steps=20,
        warmup_steps=500,
        weight_decay=0.01
    )

    data = prepare_data()

    # define the trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['eval'],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

if __name__ == '__main__':
  wandb.login()
  parse_args()
  set_seed(config.seed)
  set_vars()
  train(config)
