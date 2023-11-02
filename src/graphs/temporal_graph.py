from src.feature_extraction.roberta import RobertaInference
from src.feature_extraction.bert import BertInference
from src.feature_extraction.word2vec import Word2VecInference
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


# node features:
# - embeddings: the embeddings of the word node
# - frequency: the frequency of the word node in the dataset
# - self-similarity: the similarity of the word node to itself


# edges:
# - target node -> similar node
# - target node -> context node
# - similar node -> similar node
# - similar node -> context node
# - context node -> context node

# edge features:
# - edge_type: the type of the edge (target-similar, target-context, similar-similar, similar-context, context-context)
# - edge_weight: the weight of the edge (similarity or PMI between the nodes)
# - label: 
    # - similarity: the similarity between similar nodes
    # - mutual information: the mutual information between context nodes




class Nodes:
    def __init__(
            self,
            target_word: str,
            dataset: List[str],
            level: int,
            k: int,
            c: int,
            word2vec_model_path: str,
            mlm_model_path: str,
            mlm_model_type: str = 'roberta'
            ):
        
        self.target_word = target_word
        self.dataset = dataset
        self.k = k
        self.c = c
        self.level = level
        self.word2vec = Word2VecInference(word2vec_model_path)
        self.mlm = RobertaInference(mlm_model_path) if mlm_model_type == 'roberta' else BertInference(mlm_model_path)
    

    def _get_similar_nodes(self, word) -> List[str]:
        similar_nodes = []
        for sentence in self.dataset:
            similar_nodes += self.mlm.get_top_k_words(word, sentence, self.k)
        return list(set(similar_nodes))

    def _get_context_nodes(self, word) -> List[str]:
        context_nodes = []
        for sentence in self.dataset:
            context_nodes += self.word2vec.get_top_k_words(word, sentence, self.c)
        return list(set(context_nodes))
    
    def get_nodes(self) -> List[str]:
        nodes = {'target_node': self.target_word, 'similar_nodes': [], 'context_nodes': []}
        for level in range(self.level):
            if level == 0:
                similar_nodes = self._get_similar_nodes(self.target_word)
                context_nodes = self._get_context_nodes(self.target_word)

                nodes['similar_nodes'].append(similar_nodes)
                nodes['context_nodes'].append(context_nodes)

            else:
                similar_nodes = []
                context_nodes = []
                for word in nodes['similar_nodes'][level-1]:
                    similar_nodes += self._get_similar_nodes(word)
                    context_nodes += self._get_context_nodes(word)
                
                nodes['similar_nodes'].append(similar_nodes)
                nodes['context_nodes'].append(context_nodes)
                
        return nodes




class Edges:
    pass






class Graph:
    def __init__(self) -> None:
        self.graph = {
            'nodes': [],
            'node_features': [],
            'edges': [],
            'edge_features': []
        }
    
    









class TemporalGraph:
    pass



