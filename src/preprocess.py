import xmltodict
import json
import os
from glob import glob

from utils.utils import read_xml

class PREPROCESS():

    def __init__(self, path='../data/articles_raw_data/'):
        self.path = path

        self.file_paths = glob(os.path.join(self.path, '*.xml'))
        self.time_periods = sorted([i.split('/')[-1].split('.')[1] for i in self.file_names])

        self.periods = len(self.time_periods)
        self.articles = {}

    def 
    





if __name__ == '__main__':
    preprocess = PREPROCESS()
    print(*preprocess.time_periods)