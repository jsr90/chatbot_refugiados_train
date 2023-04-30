from sklearn.model_selection import train_test_split
from datasets import Dataset
import os
import json
import numpy as np
import random
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from functools import partial


def transform_squad(path) -> dict:
    """
    Transform SQuAD data to desired format.

    Args:
        data (dict): Original SQuAD data.

    Returns:
        dict: Transformed SQuAD data.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)['data']
    # Initialize new dictionary
    ret = {"data": []}

    # Loop through original data
    for d in data:
        for p in d['paragraphs']:
            context = p['context']
            for qas in p['qas']:
                id = qas['id']
                question = qas['question']
                answers = {'answer_start': qas['answers'][0]['answer_start'], 'text': [
                    qas['answers'][0]['text']]}
                a = {"answers": answers, "context": context,
                     "id": id, "question": question}
                ret['data'].append(a)

    return ret


def get_train_test_val(path, test_size, seed):
    """
    Split data into train and test.

    Args:
        path (str): relative path to SQuAD dataset.
        test_size(float): size of test partition.
        seed(float): seed to ensure reproducibility.

    Returns:
        dict: dict with both train and test datasets.
    """
    dataset = transform_squad(path)['data']
    train, test = train_test_split(
        dataset, test_size=test_size, random_state=seed)
    train_dataset = Dataset.from_list(train)
    test_dataset = Dataset.from_list(test)
    dic = {'train': train_dataset, 'test': test_dataset}
    return dic


def set_vars(config) -> None:
    """
    Set environment variables for wandb configuration.
    """
    os.environ["WANDB_DIR"] = config.dir_path
    os.environ["WANDB_ENTITY"] = config.wandb_entity
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_LOG_MODEL"] = config.wandb_log_model
    os.environ["WANDB_WATCH"] = config.wandb_watch


def get_strategy(i: int) -> str:
    """
    Get evaluation strategy based on the given index.
    Args:
        i (int): Index of the evaluation strategy.
    Returns:
        str: The evaluation strategy (either "no", "steps", or "epoch").
    """
    evaluation_strategies = ["no", "steps", "epoch"]
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


def preprocess_function(examples: dict, model_name) -> dict:
    """
    Preprocess the input data using the tokenizer.
    Args:
        examples (dict): Dictionary containing raw data.
    Returns:
        dict: Dictionary containing the preprocessed data.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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


def prepare_data(path, test_size, seed, model_name) -> dict:
    """
    Load and preprocess the dataset.
    Returns:
        dict: Dictionary containing the tokenized dataset.
    """
    dataset = get_train_test_val(path, test_size, seed)
    tokenized_data = {}
    for split in dataset:
        tokenized_data[split] = dataset[split].map(partial(
            preprocess_function, model_name=model_name), batched=True, remove_columns=dataset['train'].column_names)
    return tokenized_data


def train(config) -> None:
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
        save_strategy=get_strategy(config.evaluation_strategy),
        evaluation_strategy=get_strategy(config.evaluation_strategy),
        eval_steps=20,
        warmup_steps=500,
        weight_decay=0.01,
        load_best_model_at_end=True
    )
    # get train and test datasets from given path
    data = prepare_data(config.path, config.test_size, config.seed, config.model_name)

    # define the trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("Completed.")