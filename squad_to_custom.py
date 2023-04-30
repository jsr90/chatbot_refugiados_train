import json, argparse
from fastai.vision.all import SimpleNamespace

# Define default configuration
config = SimpleNamespace(
    framework="fastai",
    path="./data.json",
    save_dir="./"
)

def parse_args() -> None:
    """
    Override default arguments with the ones provided by the user.
    """
    # Create argument parser
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')

    # Define command-line arguments
    argparser.add_argument('--path', type=str, default=config.path, help='path to SQuAD file')
    argparser.add_argument('--save_dir', type=str, default=config.save_dir, help='save directory')

    # Parse command-line arguments
    args = argparser.parse_args()

    # Update configuration with command-line arguments
    vars(config).update(vars(args))
    return

def open_file(path: str) -> dict:
    """
    Open JSON file and extract data from it.

    Args:
        path (str): Path to JSON file.

    Returns:
        dict: JSON data.
    """
    with open(path, encoding="utf-8") as f:
        return(json.load(f)['data'])

def transform_squad(data: dict) -> dict:
    """
    Transform SQuAD data to desired format.

    Args:
        data (dict): Original SQuAD data.

    Returns:
        dict: Transformed SQuAD data.
    """
    # Initialize new dictionary
    ret = {"data": []}

    # Loop through original data
    for d in data:
        for p in d['paragraphs']:
            context = p['context']
            for qas in p['qas']:
                id = qas['id']
                question = qas['question']
                answers = {'answer_start' : qas['answers'][0]['answer_start'], 'text': [qas['answers'][0]['text']]}
                a = {"answers": answers, "context": context, "id": id, "question": question}
                ret['data'].append(a)

    return ret

def save_data(data: dict, path: str) -> None:
    """
    Save transformed data to file.

    Args:
        data (dict): Transformed data.
        path (str): Path to save file.
    """
    with open(path+'data_t.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    # Parse command-line arguments
    parse_args()

    # Open and transform data
    data_t = transform_squad(open_file(config.path))

    # Save transformed data to file
    save_data(data_t, config.save_dir)
