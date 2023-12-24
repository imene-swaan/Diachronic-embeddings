from re import L
from pydantic import BaseModel, field_validator
from typing import Dict, Optional, Literal, List, Union
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


    

class GraphNodes(BaseModel):
    """
    This class is used to represent the nodes of the word graph.

    Attributes:
        similar_nodes (Optional, Dict[str, List[str]]): dictionary of similar nodes for each node of the graph. Default: None
        context_nodes (Optional, Dict[str, List[str]]): dictionary of context nodes for each node of the graph. Default: None

    """
    similar_nodes: Optional[Dict[str, List[str]]] = None 
    context_nodes: Optional[Dict[str, List[str]]] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator('similar_nodes')
    def check_similar_nodes(cls, v):
        if not isinstance(v, dict):
            raise ValueError('similar_nodes must be a dictionary')
        if not isinstance(list(v.values())[0], list):
            raise ValueError('similar_nodes values must be a list')
        if not isinstance(list(v.values())[0][0], str):
            raise ValueError('similar_nodes values must be a list of strings')
        if not isinstance(list(v.keys())[0], str):
            raise ValueError('similar_nodes keys must be a string')
        return v

    
    @field_validator('context_nodes')
    def check_context_nodes(cls, v):
        if not isinstance(v, dict):
            raise ValueError('context_nodes must be a dictionary')
        if not isinstance(list(v.values())[0], list):
            raise ValueError('context_nodes values must be a list')
        if not isinstance(list(v.values())[0][0], str):
            raise ValueError('context_nodes values must be a list of strings')
        if not isinstance(list(v.keys())[0], str):
            raise ValueError('context_nodes keys must be a string')
        return v
    
   

class TargetWords(BaseModel):
    """
    This class is used to represent the target words of the word graph.
    """
    words : Union[str, List[str], Dict[str, List[str]]]

    class Config:
        arbitrary_types_allowed = True
    
    @field_validator('words')
    def check_words(cls, v):
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if not isinstance(v[0], str):
                raise ValueError('words must be a list of strings')
            return v
        if isinstance(v, dict):
            if not isinstance(list(v.values())[0], list):
                raise ValueError('words values must be a list')
            if not isinstance(list(v.values())[0][0], str):
                raise ValueError('words values must be a list of strings')
            if not isinstance(list(v.keys())[0], str):
                raise ValueError('words keys must be a string')
            return v
        raise ValueError('words must be a string, a list of strings or a dictionary')
    


class CenteredGraph(BaseModel):
    """
    This class is used to represent the centered word graph of a target word.
    """
    target_word: str

    class Config:
        arbitrary_types_allowed = True
    
    @field_validator('target_word')
    def check_target_word(cls, v):
        if not isinstance(v, str):
            raise ValueError('target_word must be a string')
        return v

class CircularGraph(BaseModel):
    """
    This class is used to represent the circular word graph of the list of target words.
    """
    target_word: List[str]

    class Config:
        arbitrary_types_allowed = True
    
    @field_validator('target_word')
    def check_target_word(cls, v):
        if not isinstance(v, list):
            raise ValueError('target_word must be a list')
        if not isinstance(v[0], str):
            raise ValueError('target_word must be a list of strings')
        return v

class HierarchicalGraph(BaseModel):
    """
    This class is used to represent the hierarchical word graph of the list of target words.
    """
    target_word: Dict[str, List[str]]

    class Config:
        arbitrary_types_allowed = True
    
    @field_validator('target_word')
    def check_target_word(cls, v):
        if not isinstance(v, dict):
            raise ValueError('target_word must be a dictionary')
        if not isinstance(list(v.values())[0], list):
            raise ValueError('target_word values must be a list')
        if not isinstance(list(v.values())[0][0], str):
            raise ValueError('target_word values must be a list of strings')
        if not isinstance(list(v.keys())[0], str):
            raise ValueError('target_word keys must be a string')
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




