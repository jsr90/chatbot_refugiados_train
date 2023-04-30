from utils import set_seed, set_vars, train
from fastai.vision.all import SimpleNamespace
import argparse
import wandb

# Define default configuration for the experiment
config = SimpleNamespace(
    framework="fastai",
    dir_path="./",
    path="./data/data.json",
    wandb_entity="jesus-saturdays/saturdays",
    wandb_project="chatbot_refugees",
    seed=42,
    wandb_log_model="true",
    wandb_watch="true",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy=2,
    model_name="timpal0l/mdeberta-v3-base-squad2",
    num_train_epochs=25,
    test_size=0.2
)


def parse_args(config) -> None:
    """
    Override default arguments with the ones provided by the user.
    """
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument(
        '--seed', type=int, default=config.seed, help='random seed')
    argparser.add_argument('--dir_path', type=str,
                           default=config.dir_path, help='path to directory')
    argparser.add_argument(
        '--path', type=str, default=config.path, help='relative path to SQuAD file')
    argparser.add_argument('--wandb_entity', type=str,
                           default=config.wandb_entity, help='W&B entity name')
    argparser.add_argument('--wandb_project', type=str,
                           default=config.wandb_project, help='W&B project name')
    argparser.add_argument('--wandb_log_model', type=str, default=config.wandb_log_model,
                           help='save your trained model checkpoint to wandb')
    argparser.add_argument('--wandb_watch', type=str,
                           default=config.wandb_watch, help='turn off watch to log faster')
    argparser.add_argument('--train_batch_size', type=str,
                           default=config.per_device_train_batch_size, help='per_device_train_batch_size')
    argparser.add_argument('--eval_batch_size', type=str,
                           default=config.per_device_eval_batch_size, help='per_device_eval_batch_size')
    argparser.add_argument('--evaluation_strategy', type=int,
                           default=config.evaluation_strategy, help='0:no, 1:steps, 2:epoch')
    argparser.add_argument('--num_train_epochs', type=int,
                           default=config.num_train_epochs, help='number of epochs during training')
    argparser.add_argument('--test_size', type=float,
                           default=config.test_size, help='size of test split. Default 0.2')
    args = argparser.parse_args()
    vars(config).update(vars(args))
    return


if __name__ == '__main__':
    wandb.login()  # Call this function before setting up env variables to avoid login errors
    parse_args(config)
    set_seed(config.seed)
    set_vars(config)
    train(config)
    wandb.finish()
