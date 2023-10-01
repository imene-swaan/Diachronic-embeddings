import tomli
from pydantic import BaseModel
from typing import List

class Experiment(BaseModel):
    name: str
    description: str

class Dataset(BaseModel):
    name: str
    periods: List[int]


class Config(BaseModel):
    experiment: Experiment
    dataset: Dataset

    
def get_config(config_path: str) -> dict:
    """
    Read in a config file and return a dictionary.

    Args:
        config_path (str): The path to the config file.

    Returns:
        dict: The dictionary.
    """
    with open(config_path, "rb") as f:
        return tomli.load(f)



def main(config: Config):
    print(config['experiment']['name'])
    print('periods: ', config['dataset']['periods'])

    




    





if __name__ == "__main__":
    config = get_config('config.toml')
    main(config)











