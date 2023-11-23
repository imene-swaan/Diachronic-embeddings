from pydantic import BaseModel, field_validator
from typing import Dict
import numpy as np

class WordGraph(BaseModel):
    index: Dict[str, Dict]
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @field_validator('index')
    def check_index(cls, v):
        if not isinstance(v, dict):
            raise ValueError('index must be a dictionary')
        if 'key_to_index' not in v or 'index_to_key' not in v:
            raise ValueError('index must contain both key_to_index and index_to_key dictionaries')
        return v
    
    @field_validator('node_features')
    def check_node_features(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('node_features must be a numpy.ndarray')
        return v
    
    @field_validator('edge_index')
    def check_edge_index(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('edge_index must be a numpy.ndarray')
        return v
    
    @field_validator('edge_features')
    def check_edge_features(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('edge_features must be a numpy.ndarray')
        return v
    
    @field_validator('labels')
    def check_labels(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('labels must be a numpy.ndarray')
        return v

    @field_validator('label_mask')
    def check_label_mask(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('label_mask must be a numpy.ndarray')
        return v


   
    
if __name__ == '__main__':
    g = WordGraph(
        index={
            'key_to_index': {'a': 0, 'b': 1, 'c': 2},
            'index_to_key': {0: 'a', 1: 'b', 2: 'c'}
        },
        node_features=np.array([[1, 2], [3, 4], [5, 6]]),
        edge_index=np.array([[0, 1], [1, 2]]),
        edge_features=np.array([[0.1, 0.2], [0.3, 0.4]]),
        labels=np.array([0, 1, 2]),
        label_mask=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )
    
    # index={
    #         'key_to_index': {'a': 0, 'b': 1, 'c': 2},
    #         'index_to_key': {0: 'a', 1: 'b', 2: 'c'}
    #     }
    
    # print(index['key_to_index']['a'])
    # print(index['index_to_key'][0])

    print(g.index['index_to_key'][0])




