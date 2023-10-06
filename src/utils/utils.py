import yaml
import tomli
import random

def read_toml(config_path: str) -> dict:
    """
    Read in a config file and return a dictionary.

    Args:
        config_path (str): The path to the config file.

    Returns:
        dict: The dictionary.
    """
    with open(config_path, "rb") as f:
        return tomli.load(f)

def read_yaml(file_path: str) -> dict:
    """
    Read in a yaml file and return a dictionary.

    Args:
        file_path (str): The path to the yaml file.

    Returns:
        dict: The dictionary.
    """
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def read_txt(file_path: str) -> list:
    """
    Read in a txt file and return a list of lines.

    Args:
        file_path (str): The path to the txt file.

    Returns:
        list: A list of lines.
    """

    with open(file_path) as f:
        return f.readlines()
    

def train_test_split(data: list, test_ratio=0.2, random_seed=None):
    """
    Split the data into train and test sets.

    Args:
        data (list): The data to split.
        test_ratio (float): The ratio of the test set.
        random_seed (int): The random seed.

    Returns:
        tuple: A tuple of the train and test sets.
    """
    
    if random_seed:
        random.seed(random_seed)
    data_copy = data[:]
    random.shuffle(data_copy)
    split_idx = int(len(data_copy) * (1 - test_ratio))
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]
    return train_data, test_data


def sample_data(data: list, sample_size: int, random_seed=None):
    """
    Sample data.

    Args:
        data (list): The data to sample.
        sample_size (int): The size of the sample.
        random_seed (int): The random seed.

    Returns:
        list: The sampled data.
    """
    
    if random_seed:
        random.seed(random_seed)
    data_copy = data[:]
    random.shuffle(data_copy)
    return data_copy[:sample_size]
