from hmac import new
from re import T
from click import Option
from regex import E
from semantics.feature_extraction.roberta import RobertaInference
from semantics.feature_extraction.bert import BertInference
from semantics.feature_extraction.word2vec import Word2VecInference
from typing import List, Union, Dict, Optional, Tuple
import numpy as np
from semantics.utils.utils import count_occurence, most_frequent
import tqdm
from semantics.utils.components import  GraphNodes, TargetWords, GraphIndex
from pydantic import ValidationError
import json

def print_dict_as_json(dict: Dict[str, List[str]]):
    print(json.dumps(dict, indent=4), '\n')
class Nodes:
    def __init__(
            self,
            target: Union[str, List[str], Dict[str, List[str]]],
            ) -> None:

        try:
            TargetWords(words=target)
        
        except ValidationError:
            raise ValueError('The target word must be a string, a list of strings, or a dictionary. Check the TargetWords class for more information.')


        else:
            if isinstance(target, str):
                self.graph_nodes = GraphNodes()
                self.initial_target = [target]


            elif isinstance(target, list):
                similar_nodes= {}
                for w in target:
                    similar_nodes[w] = [word for word in target if word != w]
                self.graph_nodes = GraphNodes(similar_nodes= similar_nodes)
                self.initial_target = target

            else:
                self.graph_nodes = GraphNodes(similar_nodes= target)
                self.initial_target = list(set(sum(list(target.values()), [])))

            self.target = self.initial_target.copy()
            

    def get_nodes(
            self,
            dataset: Optional[List[str]] = None,
            level: int = 0,
            k: int = 3,
            c: int = 2,
            word2vec_model: Optional[Word2VecInference] = None,
            mlm_model: Optional[Union[RobertaInference, BertInference]] = None,
            keep_k: Optional[Dict[int, Tuple[int, int]]] = None
            ) -> GraphNodes:
        

        if keep_k is None:
            keep_k = {0: (6, 2)}
            for i in range(1, level):
                keep_k[i] = (2, 1)
        
        else:
            if (level - len(list(keep_k.keys()))) > 0:
                repeat = keep_k[len(list(keep_k.keys()))-1]
                for i in range(len(list(keep_k.keys())), level):
                    keep_k[i] = repeat

        
        if level < 0:
            raise ValueError('The level must be greater than or equal to 0.')
        

        if level == 0:
            if len(self.target) == 1:
                raise ValueError('The level must be greater than 0 for a single target word.')
            
            return self.graph_nodes

        
        if level > 0:
            if dataset is None:
                raise ValueError('The dataset must be provided when the level is greater than 0.')
            
            if word2vec_model is None and mlm_model is None:
                raise ValueError('The word2vec or the mlm model must be provided when the level is greater than 0.')
        

            nb = NodesBuilder(
                dataset=dataset,
                k=k,
                c=c,
                word2vec_model=word2vec_model,
                mlm_model=mlm_model,
            )

            for i in range(level):
                similar_nodes, context_nodes = nb.add_level(target=self.target, keep_k=keep_k[i])
                
                new_target = []
                if similar_nodes is not None:
                    if self.graph_nodes.similar_nodes is None:
                        self.graph_nodes.similar_nodes = similar_nodes
                        new_target += sum(list(similar_nodes.values()), [])
                    else:
                        for key, value in similar_nodes.items():
                            new_target += value
                            if key in self.graph_nodes.similar_nodes.keys():
                                self.graph_nodes.similar_nodes[key] += value
                                self.graph_nodes.similar_nodes[key] = list(set(self.graph_nodes.similar_nodes[key]))
                            else:
                                self.graph_nodes.similar_nodes[key] = value
                else:
                    if self.graph_nodes.context_nodes is None:
                        self.graph_nodes.context_nodes = self.graph_nodes.similar_nodes
                        self.graph_nodes.similar_nodes = None

                if context_nodes is not None:
                    if self.graph_nodes.context_nodes is None:
                        self.graph_nodes.context_nodes = context_nodes
                        new_target += sum(list(context_nodes.values()), [])
                    else:
                        for key, value in context_nodes.items():
                            new_target += value
                            if key in self.graph_nodes.context_nodes.keys():
                                self.graph_nodes.context_nodes[key] += value
                                self.graph_nodes.context_nodes[key] = list(set(self.graph_nodes.context_nodes[key]))
                            else:
                                self.graph_nodes.context_nodes[key] = value
                        
                new_target = list(set(new_target))
                self.target = [word for word in new_target if word not in self.target]
            return self.graph_nodes


    def get_node_features(
            self,
            dataset: List[str],
            mlm_model: Union[RobertaInference, BertInference]
            ) -> Tuple[GraphIndex, np.ndarray, np.ndarray]:
        """
        """

        all_words = []
        node_types = []
        frequencies = []
        embeddings = []

        all_words_count = count_occurence(dataset)

        for node in self.initial_target:
            all_words.append(node)
            node_types.append(1)
            frequencies.append(count_occurence(dataset, node) / all_words_count)
            embeddings.append(mlm_model.get_embedding(main_word=node).mean(axis=0))
        
        if self.graph_nodes.similar_nodes is not None:
            for node_list in self.graph_nodes.similar_nodes.values():
                for node in node_list:
                    if node in all_words:
                        continue

                    all_words.append(node)
                    node_types.append(2)
                    frequencies.append(count_occurence(dataset, node) / all_words_count)
                    embeddings.append(mlm_model.get_embedding(main_word=node).mean(axis=0))
            
        if self.graph_nodes.context_nodes is not None:
            for node_list in self.graph_nodes.context_nodes.values():
                for node in node_list:
                    if node in all_words:
                        continue

                    all_words.append(node)
                    node_types.append(3)
                    frequencies.append(count_occurence(dataset, node) / all_words_count)
                    embeddings.append(mlm_model.get_embedding(main_word=node).mean(axis=0))
            
        index_to_key = {idx: word for idx, word in enumerate(all_words)}
        key_to_index = {word: idx for idx, word in enumerate(all_words)}

        del all_words
        embeddings = np.array(embeddings)
        node_features = np.stack([node_types, frequencies]).T

        index = GraphIndex(index_to_key=index_to_key, key_to_index=key_to_index)

        assert len(list(index_to_key.keys())) == node_features.shape[0] == embeddings.shape[0]
        return index, node_features, embeddings
        

class NodesBuilder:
    """
    """
    def __init__(
            self,
            dataset: List[str],
            k: int,
            c: int,
            word2vec_model: Optional[Word2VecInference] = None,
            mlm_model: Optional[Union[RobertaInference, BertInference]] = None,
            ):
        
    

        
        self.dataset = dataset
        self.k = k
        self.c = c
        self.word2vec = word2vec_model
        self.mlm = mlm_model

   


    def add_level(self, target: List[str], keep_k: Tuple[int, int]):
        """
        """
        if self.mlm is not None and self.word2vec is not None:
            similar_nodes = self._get_similar_nodes(target, keep_k= keep_k[0])
            context_nodes = self._get_context_nodes(target, keep_k= keep_k[1])
            return similar_nodes, context_nodes
        
        elif self.mlm is not None:
            similar_nodes = self._get_similar_nodes(target, keep_k= keep_k[0])
            context_nodes = None
            return similar_nodes, context_nodes
        
        else:
            similar_nodes = None
            context_nodes = self._get_context_nodes(target, keep_k= keep_k[0])
            return similar_nodes, context_nodes



    def _get_similar_nodes(
            self, 
            word: Union[str, List[str]],
            keep_k: int = 50
            ) -> Dict[str, List[str]]:
        """
        """

        if isinstance(word, str):
            word = [word]
        
        print(f'Getting the similar nodes for the words: {word} ...')

        progress_bar = tqdm.tqdm(total=len(self.dataset))
        similar_nodes = {w: [] for w in word}
        
        for sentence in self.dataset:
            for w in word:
                similar_nodes[w] += self.mlm.get_top_k_words(main_word=w, doc = sentence, k= self.k)
            progress_bar.update(1)

        for w in word:
            if len(similar_nodes[w]) > 0:
                similar_nodes[w], _ = most_frequent(similar_nodes[w], keep_k)
            else:
                del similar_nodes[w]
        return similar_nodes

               

    def _get_context_nodes(
            self, 
            word: Union[str, List[str]],
            keep_k: int = 50
            ) -> Dict[str, List[str]]:
        """
        """
        if isinstance(word, str):
            word = [word]
        
        context_nodes = {}
        print(f'Getting the context nodes for the words: {word} ...')
        for w in word:
            k_words, _ = self.word2vec.get_top_k_words(w, self.c)
            
            if len(k_words) > 0:
                context_nodes[w] = k_words[:keep_k]
        return context_nodes
