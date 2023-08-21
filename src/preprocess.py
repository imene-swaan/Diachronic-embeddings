import xmltodict
import json
import yaml
import os
from glob import glob

from utils.utils import *

class PREPROCESS():

    def __init__(self, path='../data/articles_raw_data/', config_path='config.yaml'):
        self.path = path

        self.file_paths = glob(os.path.join(self.path, '*.xml'))
        self.time_periods = sorted([i.split('/')[-1].split('.')[1] for i in self.file_paths])

        self.periods = len(self.time_periods)
        self.articles = [get_articles(self.file_paths[i]) for i in range(self.periods)]
        self.sentences = [get_sentences(self.articles[i]) for i in range(self.periods)]
        
        with open(config_path, "rb") as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
    
    
        if self.configs['save_sentences']:
            save_sentences(self.sentences, self.time_periods)





if __name__ == '__main__':
    preprocess = PREPROCESS()
    print(*preprocess.time_periods)
