from pydantic import BaseModel, field_validator
from typing import Dict, Optional, List, Union
import numpy as np
from sympy import Li



    

class GraphNodes(BaseModel):
    """
    This class is used to represent the nodes of the word graph.

    Attributes:
        similar_nodes (Optional, Dict[str, List[str]]): dictionary of similar nodes for each node of the graph. Default: None
        context_nodes (Optional, Dict[str, List[str]]): dictionary of context nodes for each node of the graph. Default: None

    """
    similar_nodes: Optional[Dict[str, List[str]]] = None
    context_nodes: Optional[Dict[str, List[str]]] = None
    target_nodes: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True

  
    @field_validator('similar_nodes', 'context_nodes')
    def check_nodes(cls, v, field):
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError(f'{field.field_name} must be a dictionary')
            for key, val in v.items():
                if not isinstance(key, str):
                    raise ValueError(f'All keys in {field.field_name=} must be strings')
                if not isinstance(val, list):
                    raise ValueError(f'All values in {field.field_name=} must be lists')
                # if len(val) == 0:
                #     raise ValueError(f'All lists in {field.field_name} must be non-empty. the key: ({key}) has an empty list')
                if not all(isinstance(item, str) for item in val):
                    raise ValueError(f'All elements in the lists of {field.field_name=} must be strings')
        return v
    
    @field_validator('target_nodes')
    def check_target_nodes(cls, v):
        if v is not None:
            if not isinstance(v, list):
                raise ValueError('target_nodes must be a list')
            # if len(v) == 0:
            #     raise ValueError('target_nodes cannot be an empty list')
            if not all(isinstance(item, str) for item in v):
                raise ValueError('All elements in the list must be strings')
        return v


  

class GraphIndex(BaseModel):
    index_to_key: Dict[int, str]
    key_to_index: Dict[str, int]

    class Config:
        arbitrary_types_allowed = True

    @field_validator('index_to_key')
    def check_index_to_key(cls, v):
        if not isinstance(v, dict):
            raise ValueError('index_to_key must be a dictionary')
        if not all(isinstance(key, int) for key in v.keys()):
            raise ValueError('All keys in index_to_key must be integers')
        if not all(isinstance(val, str) for val in v.values()):
            raise ValueError('All values in index_to_key must be strings')
        return v

    @field_validator('key_to_index')
    def check_key_to_index(cls, v, values):
        if not isinstance(v, dict):
            raise ValueError('key_to_index must be a dictionary')
        if not all(isinstance(key, str) for key in v.keys()):
            raise ValueError('All keys in key_to_index must be strings')
        if not all(isinstance(val, int) for val in v.values()):
            raise ValueError('All values in key_to_index must be integers')
        
        if len(list(v.keys())) != len(list(values.data['index_to_key'].keys())):
            raise ValueError('key_to_index and index_to_key must have the same number of elements')

        if not all(key in list(values.data['index_to_key'].values()) for key in v.keys()):
            raise ValueError('All keys in key_to_index must be in index_to_key values')

        if not all(val in list(values.data['index_to_key'].keys()) for val in v.values()):
            raise ValueError('All values in key_to_index must be in index_to_key keys')
        
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
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError('words cannot be an empty list')
            if not all(isinstance(item, str) for item in v):
                raise ValueError('All elements in the list must be strings')
            return v
        elif isinstance(v, dict):
            if not all(isinstance(key, str) for key in v.keys()):
                raise ValueError('All keys in the dictionary must be strings')
            if not all(isinstance(val, list) and all(isinstance(item, str) for item in val) for val in v.values()):
                raise ValueError('All values in the dictionary must be lists of strings')
            return v
        raise ValueError('words must be a string, a list of strings, or a dictionary')



class WordGraph(BaseModel):
    index: GraphIndex
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @field_validator('index')
    def check_index(cls, v):
        if not isinstance(v, GraphIndex):
            raise ValueError('index must be a GraphIndex object. Use GraphIndex class to create a GraphIndex object')
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
    # g = WordGraph(
    #     index={
    #         'key_to_index': {'a': 0, 'b': 1, 'c': 2},
    #         'index_to_key': {0: 'a', 1: 'b', 2: 'c'}
    #     },
    #     node_features=np.array([[1, 2], [3, 4], [5, 6]]),
    #     edge_index=np.array([[0, 1], [1, 2]]),
    #     edge_features=np.array([[0.1, 0.2], [0.3, 0.4]]),
    #     labels=np.array([0, 1, 2]),
    #     label_mask=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # )
    
    # index={
    #         'key_to_index': {'a': 0, 'b': 1, 'c': 2},
    #         'index_to_key': {0: 'a', 1: 'b', 2: 'c'}
    #     }
    
    # print(index['key_to_index']['a'])
    # print(index['index_to_key'][0])

    # print(g.index['index_to_key'][0])

    data = GraphNodes()
    data.similar_nodes = {'a': ['b', 'c'], 'b': ['a', 'c'], 'c': ['a', 'b']}

    print(data.similar_nodes['a'])

    # index_to_key = {0: 'a', 1: 'b', 2: 'c'}
    # key_to_index = {'a': 0, 'b': 1, 'c': 2}
    # c = GraphIndex(index_to_key=index_to_key, key_to_index=key_to_index)
    # print(dict(c))


