from semantics.feature_extraction.roberta import RobertaInference
from semantics.feature_extraction.bert import BertInference
from semantics.feature_extraction.word2vec import Word2VecInference
from typing import List, Union, Dict, Optional, Tuple
import numpy as np
from semantics.utils.utils import count_occurence, most_frequent
import tqdm
from semantics.utils.components import  GraphNodes, TargetWords, GraphIndex
from pydantic import ValidationError



class Nodes:
    """
    """
    def __init__(
            self,
            target: Union[str, List[str], Dict[str, List[str]]],
            dataset: List[str],
            level: int,
            k: int,
            c: int,
            word2vec_model: Word2VecInference,
            mlm_model: Union[RobertaInference, BertInference],
            keep_k: Optional[Dict[int, Tuple[int, int]]] = None
            ):
        
    

        
        self.dataset = dataset
        self.k = k
        self.c = c
        self.level = level
        self.word2vec = word2vec_model
        self.mlm = mlm_model

        if keep_k is None:
            self.keep_k = {0: (6, 2)}
            for i in range(1, self.level):
                self.keep_k[i] = (2, 1)
        
        else:
            self.keep_k = keep_k
            if (self.level - len(list(keep_k.keys()))) > 0:
                repeat = keep_k[len(list(keep_k.keys()))-1]
                for i in range(len(list(keep_k.keys())), self.level):
                    self.keep_k[i] = repeat
                    

        try:
            TargetWords(words=target)
        
        except ValidationError:
            raise ValueError('The target word must be a string, a list of strings, or a dictionary. Check the TargetWords class for more information.')
        
        else:
            if isinstance(target, str):
                print('The target word is a string...')
                if self.level < 1:
                    raise ValueError('The level must be greater than 0 for a single target word.')
                
                self.graph_nodes = GraphNodes(target_nodes= [target])
                print('Graph nodes: ', self.graph_nodes)
                self.target = [target]

            elif isinstance(target, list):
                print('The target word is a list of strings...')
                if self.level < 0:
                    raise ValueError('The level must be greater than or equal to 0 for a list of target words.')
                similar_nodes= {}
                for w in target:
                    similar_nodes[w] = [word for word in target if word != w]
                self.graph_nodes = GraphNodes(target_nodes= target, similar_nodes= similar_nodes)
                print('Graph nodes: ', self.graph_nodes)
                self.target = target

            else:
                print('The target word is a dictionary...')
                if self.level < 0:
                    raise ValueError('The level must be greater than or equal to 0 for a dictionary of target words.')
                target_nodes = sum(list(target.values()), [])
                target_nodes = list(set(target_nodes))
                # ignore the target nodes that are already in the similar nodes
                target_nodes = [word for word in target_nodes if word not in list(target.keys())]
                self.graph_nodes = GraphNodes(target_nodes= target_nodes, similar_nodes= target)
                print('Graph nodes: ', self.graph_nodes)
                self.target = list(target.keys())
        
    
    def generate_nodes(self) -> None:
        for level in range(self.level):
            print(f'Adding the nodes of level {level}...')
            similar_nodes, context_nodes = self._add_level(level)
            target_nodes = []
            
            if self.graph_nodes.similar_nodes is None:
                self.graph_nodes.similar_nodes = similar_nodes
                self.graph_nodes.context_nodes = context_nodes

                target_nodes = sum(list(similar_nodes.values()), []) + sum(list(context_nodes.values()), [])
                
            
            else:
                for key in similar_nodes.keys():
                    target_nodes += similar_nodes[key]
                    if key in self.graph_nodes.similar_nodes.keys():
                        self.graph_nodes.similar_nodes[key] += similar_nodes[key]
                        self.graph_nodes.similar_nodes[key] = list(set(self.graph_nodes.similar_nodes[key]))
                    else:
                        self.graph_nodes.similar_nodes[key] = similar_nodes[key]
                
                if self.graph_nodes.context_nodes is None:
                    self.graph_nodes.context_nodes = context_nodes
                    target_nodes += sum(list(context_nodes.values()), [])
                
                else:
                    for key in context_nodes.keys():
                        target_nodes += context_nodes[key]
                        if key in self.graph_nodes.context_nodes.keys():
                            self.graph_nodes.context_nodes[key] += context_nodes[key]
                            self.graph_nodes.context_nodes[key] = list(set(self.graph_nodes.context_nodes[key]))
                        else:
                            self.graph_nodes.context_nodes[key] = context_nodes[key]
            
            target_nodes = [word for word in target_nodes if word not in self.graph_nodes.target_nodes]
            self.graph_nodes.target_nodes = list(set(target_nodes))


   

    def _add_level(self, level: int):
        similar_nodes = self._get_similar_nodes(self.graph_nodes.target_nodes, keep_k=self.keep_k[level][0])
        context_nodes = self._get_context_nodes(self.graph_nodes.target_nodes, keep_k=self.keep_k[level][1])

        return similar_nodes, context_nodes
        
    def get_node_features(self) -> Tuple[GraphIndex, np.ndarray, np.ndarray]:
        """
        """
            
        words = self.target
        node_types = [1]*len(words)
        all_words_count = count_occurence(self.dataset)
        frequencies = [count_occurence(self.dataset, word)/ all_words_count for word in words]
        embeddings = [self.mlm.get_embedding(main_word=word).mean(axis=0) for word in words]

        for node_list in self.graph_nodes.similar_nodes.values():
            for node in node_list:
                if node in words:
                    continue

                words.append(node)
                node_types.append(2)
                frequencies.append(count_occurence(self.dataset, node) / all_words_count)
                embeddings.append(self.mlm.get_embedding(main_word=node).mean(axis=0))

        for node_list in self.graph_nodes.context_nodes.values():
            for node in node_list:
                if node in words:
                    continue

                words.append(node)
                node_types.append(3)
                frequencies.append(count_occurence(self.dataset, node) / all_words_count)
                embeddings.append(self.mlm.get_embedding(main_word=node).mean(axis=0))


        index_to_key = {idx: word for idx, word in enumerate(words)}
        key_to_index = {word: idx for idx, word in enumerate(words)}  

        del words
        embeddings = np.array(embeddings)
        node_features = np.stack([node_types, frequencies]).T

        index = GraphIndex(index_to_key=index_to_key, key_to_index=key_to_index)
        return index, node_features, embeddings



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
