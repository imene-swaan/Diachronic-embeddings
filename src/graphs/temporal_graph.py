from src.feature_extraction.roberta import RobertaInference
from src.feature_extraction.bert import BertInference
from src.feature_extraction.word2vec import Word2VecInference
from pydantic import BaseModel, validator, Field
from typing import List, Optional, Union


# nodes:
# level 1:
    # - context nodes: words that are used in the same context as the target word. Extracted from word2vec.
    # - similar nodes: words that are similar to the target word. Extracted from the MLM by checking the k most likely words to replace the masked target word in each sentence.
    # - target node: the target word.

# level 2:
    # - context nodes: context nodes of words in level 1
    # - similar nodes: similar nodes of words in level 1

# level 3:
    # - context nodes: context nodes of words in level 2
    # - similar nodes: similar nodes of words in level 2

# args:
    # - target word: the word to get the nodes for
    # - dataset: the sentences to get the nodes from
    # - k: the number of similar nodes to get for each occurrence of the target word
    # - c: the number of context nodes to get for the target word
    # - level: the level of the graph to get
    # - model_paths: the path to the word2vec model and the MLM model



class Nodes:
    def __init__(
            self,
            target_word: str,
            dataset: List[str],
            level: int,
            k: int,
            c: int,
            model_paths: List[str]
            ):
        
        self.target_word = target_word
        self.dataset = dataset
        self.k = k
        self.c = c
        self.level = level
        self.model_paths = model_paths


    





class Edges:
    pass






class Graph:
    def __init__(self) -> None:
        pass








class TemporalGraph:
    pass



