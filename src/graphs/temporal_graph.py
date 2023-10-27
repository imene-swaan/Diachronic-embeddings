from src.feature_extraction.roberta import RobertaInference
from src.feature_extraction.bert import BertInference
from src.feature_extraction.word2vec import Word2VecInference
from pydantic import BaseModel, validator, Field
from typing import List, Optional, Union





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
