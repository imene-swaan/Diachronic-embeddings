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
    def __init__(self):
        pass

    def get_context_nodes(
            self,
            word: str, 
            model_path: Optional[str] = None, 
            k: int = 10
            ) -> tuple(List[str], List[float]):
        """
        Get the context nodes of a given word.

        Args:
            word (str): The word to get the context nodes.
            model_path (Optional[str], optional): The path to the model. Defaults to None.
            k (int, optional): The number of context nodes to get. Defaults to 5.

        Returns:
            (context_words, similarities) (tuple(List[str], List[float])): The context nodes and the corresponding similarity scores.
        """
        word2vec = Word2VecInference(model_path)
        return word2vec.get_top_k_words(
            word=word,
            k=k
        )

    def get_similar_nodes(
            self,
            word: str,
            sentence: str,
            model_path: Optional[str] = None,
            k: int = 10
            ) -> List[str]:

        """
        Get the similar nodes of a given word.

        Args:
            word (str): The word to get the similar nodes.
            model_path (Optional[str], optional): The path to the model. Defaults to None.
            k (int, optional): The number of similar nodes to get. Defaults to 5.

        Returns:
            List[str]: The similar nodes.
        """

        MLM = RobertaInference(model_path)
        return MLM.get_top_k_words(
            word=word,
            sentence=sentence,
            k=k
        )








class Edges:
    pass

class Graph:
    pass

class TemporalGraph:
    pass



if __name__ == '__main__':
    pass
