from pydantic import BaseModel, validator
from typing import List
from src.utils.utils import read_toml as get_config
from src.data.data_loader import Loader
from src.data.data_preprocessing import PREPROCESS

class Experiment(BaseModel):
    name: str
    description: str

class Dataset(BaseModel):
    name: str
    periods: List[int]
    file_type: str
    path: str
    tag: str

class Preprocessing(BaseModel):
    skip: bool
    options: dict

    #validate that options values are boolean
    @validator('options')
    def validate_options(cls, v):
        for key in v:
            if not isinstance(v[key], bool):
                raise ValueError('options values must be boolean')
        return v

class Masked_language_model(BaseModel):
    skip: bool
    architecture: str

    tokenizer: str
    tokenizer_options: dict
    mlm_options: dict

    model = str
    train = bool
    model_options = dict

    evaluation = bool
    train_test_split = float
    evaluation_options = dict

    @validator('tokenizer_options')
    def validate_tokenizer_options(cls, v):
        for key in v:
            if key == "max_length":
                if not isinstance(v[key], int):
                    raise ValueError('max_length must be integer')
                else:
                    if v[key] > 512:
                        raise ValueError('max_length must be less than 512')
            break  
        return v
    
    @validator('mlm_options')
    def validate_mlm_options(cls, v):
        for key in v:
            if key == "mlm_probability":
                if not isinstance(v[key], float):
                    raise ValueError('mlm_probability must be float')
                else:
                    if v[key] > 1 or v[key] < 0:
                        raise ValueError('mlm_probability must be between 0 and 1')
        return v
    
    





class Config(BaseModel):
    experiment: Experiment
    dataset: Dataset
    preprocessing: Preprocessing
    Masked_language_model: Masked_language_model

    


def main(config: Config):
    print('experiment: ', config['experiment']['name'])
    print('*'*10, 'Loading data', '*'*10, '\n')

    corpora = {}

    if config['dataset']['file_type'] == 'xml':
        for period in config['dataset']['periods']:
            path = config['dataset']['path'].format(period)
            data = Loader.from_xml(path, config['dataset']['tag']).forward()
            corpora[period] = data



    
    elif config['dataset']['file_type'] == 'txt':
        for period in config['dataset']['periods']:
            path = config['dataset']['path'].format(period)
            data = Loader.from_txt(path).forward()
            corpora[period] = data

    
    else:
        raise ValueError('File type not supported')
    
    if not config['preprocessing']['skip']:
        print('*'*10, 'Preprocessing data', '*'*10, '\n')
        for period in config['dataset']['periods']:
            print('before: ', corpora[period][0][:100])
            corpora[period] = list(map(lambda x: PREPROCESS().forward(x, **config['preprocessing']['options']), corpora[period]))
            print('after: ', corpora[period][0][:100])


    if not config['Masked_language_model']['skip']:
        print('*'*10, 'Masked language modeling (Diachronic Embeddings)', '*'*10, '\n')
        
        
    


    




if __name__ == "__main__":
    config = get_config('config.toml')
    main(config)











