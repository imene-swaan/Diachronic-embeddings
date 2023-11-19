from pydantic import BaseModel, validator, dataclasses
from typing import Dict, Union
import numpy as np

from pydantic import BaseModel, validator
from typing import Dict, Union
import numpy as np

class WordGraph(BaseModel):
    index: Dict[str, Union[Dict[str, int], Dict[str, str]]]
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray

    @validator('index')
    def check_index(cls, v):
        if not isinstance(v, dict):
            raise ValueError('index must be a dictionary')
        if 'key_to_index' not in v or 'index_to_key' not in v:
            raise ValueError('index must contain both key_to_index and index_to_key')
        return v
    
    @validator('node_features')
    def check_node_features(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('node_features must be a numpy.ndarray')
        if len(v.shape) != 2:
            raise ValueError('node_features must be a 2D array')
        return v
    
    @validator('edge_index')
    def check_edge_index(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('edge_index must be a numpy.ndarray')
        if len(v.shape) != 2:
            raise ValueError('edge_index must be a 2D array')
        return v
    
    @validator('edge_features')
    def check_edge_features(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('edge_features must be a numpy.ndarray')
        if len(v.shape) != 2:
            raise ValueError('edge_features must be a 2D array')
        return v
    
    @validator('labels')
    def check_labels(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('labels must be a numpy.ndarray')
        if len(v.shape) != 1:
            raise ValueError('labels must be a 1D array')
        return v

    @validator('label_mask')
    def check_label_mask(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError('label_mask must be a numpy.ndarray')
        if len(v.shape) != 2:
            raise ValueError('label_mask must be a 2D array')
        return v


   
    







