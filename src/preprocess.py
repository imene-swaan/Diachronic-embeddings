import xmltodict
import json
import yaml
import os
from glob import glob

from utils.utils import *

class PREPROCESS():

    def __init__(self, path='../input/xml/', config_path='config.yaml'):
        self.path = path

        with open(config_path, "rb") as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)

        if str(self.configs['periods']).lower() == 'all':
            self.file_paths = glob(os.path.join(self.path, '*.xml'))
            self.time_periods = sorted([i.split('/')[-1].split('.')[1] for i in self.file_paths])
        
        else:
            self.time_periods = self.configs['periods']
            self.file_paths = [i for i in glob(os.path.join(self.path, '*.xml')) if i.split('/')[-1].split('.')[1] in self.time_periods]


        self.periods = len(self.time_periods)
        print(f'Found {self.periods} periods')

        for i, p in enumerate(self.time_periods):
            print(f'Processing {p} period')
            articles = get_articles(self.file_paths[i])
            sentences = get_sentences(articles)

            print(f'Found {len(articles)} articles')

            
            if not self.configs['Preprocessing']['skip']:
                print('Preprocessing articles')
                clean_articles = [cleantxt(a, self.configs['Preprocessing']['options']) for a in articles]

            else:
                clean_articles = articles
            

            if self.configs['save_articles']:
                print(f'Saving {len(clean_articles)} articles at ../input/articles/{p}_articles.json')
                save_texts(clean_articles, f'../input/articles/{p}_articles.txt')
            
            if self.configs['save_sentences']:
                save_texts(sentences, f'../input/sentences/{p}_sentences.txt')

            print(f'Finished processing {p} period\n\n\n')



    
        



if __name__ == '__main__':
    preprocess = PREPROCESS()
    print(*preprocess.time_periods)
