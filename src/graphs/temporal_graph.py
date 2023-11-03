from src.feature_extraction.roberta import RobertaInference
from src.feature_extraction.bert import BertInference
from src.feature_extraction.word2vec import Word2VecInference
from typing import List, Optional, Union, Dict


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
# - node_type: the type of the node (target, similar, context)
# - node_level: the level of the node in the graph
# - embeddings: the embeddings of the word node
# - frequency: the frequency of the word node in the dataset


# edge_index:
# - target node -> similar node
# - target node -> context node
# - similar node -> similar node
# - similar node -> context node
# - context node -> context node

# edge features:
# - edge_type: the type of the edge (target-similar, target-context, similar-similar, similar-context, context-context, self-loop)
# - similarity: the similarity between node embeddings in the current snapshot
# - PMI: the PMI between nodes in the current snapshot

# - labels: 
    # - similarity: the similarity between similar nodes in the next snapshot





class Nodes:
    """
    This class is used to get the nodes of the word graph.

    methods:
        __init__(self, target_word: str, dataset: List[str], level: int, k: int, c: int, word2vec_model_path: str, mlm_model_path: str, mlm_model_type: str = 'roberta')
            The constructor of the Nodes class.
        _get_similar_nodes(self, word: str) -> List[str]
            This method is used to get the similar nodes of a word.
        
        _get_context_nodes(self, word: str) -> List[str]
            This method is used to get the context nodes of a word.
        
        get_nodes(self) -> Dict[str, List[str]]
            This method is used to get the nodes of the word graph.
    """
    def __init__(
            self,
            target_word: str,
            dataset: List[str],
            level: int,
            k: int,
            c: int,
            word2vec_model: Word2VecInference,
            mlm_model: Union[RobertaInference, BertInference]
            ):
        
        """
        Args:
            target_word (str): the word to get the nodes for
            dataset (List[str]): the sentences to get the nodes from
            level (int): the level of the graph to get
            k (int): the number of similar nodes to get for each occurrence of the target word
            c (int): the number of context nodes to get for the target word
            word2vec_model_path (str): the path to the word2vec model
            mlm_model_path (str): the path to the MLM model
            mlm_model_type (str, optional): the type of the MLM model. Defaults to 'roberta'.
        """

        self.target_word = target_word
        self.dataset = dataset
        self.k = k
        self.c = c
        self.level = level
        self.word2vec = word2vec_model
        self.mlm = mlm_model
    

    def _get_similar_nodes(self, word: str) -> List[str]:
        """
        This method is used to get the similar nodes of a word using the MLM model.
        
        Args:
            word (str): the word to get the similar nodes for
            
        Returns:
            similar nodes (List[str]): the list of similar nodes of the word
        """
        similar_nodes = []
        for sentence in self.dataset:
            similar_nodes += self.mlm.get_top_k_words(word, sentence, self.k)
        return list(set(similar_nodes))

    def _get_context_nodes(self, word: str) -> List[str]:
        """
        This method is used to get the context nodes of a word using the word2vec model.

        Args:
            word (str): the word to get the context nodes for

        Returns:
            context nodes (List[str]): the list of context nodes of the word
        """
        context_nodes, _ = self.word2vec.get_top_k_words(word, self.c)
        return list(set(context_nodes))
    
    def get_nodes(self) -> Dict[str, List[str]]:
        """
        This method is used to get the nodes of the word graph (similar nodes, context nodes, and target node).

        Returns:
            nodes (Dict[str, List[str]]): the nodes of the word graph

        
        Example:
            >>> n = Nodes(
                    target_word='sentence',
                    dataset=['this is a sentence', 'this is another sentence'],
                    level=3,
                    k=2,
                    c=2,
                    word2vec_model_path='word2vec.model',
                    mlm_model_path='MLM_roberta',
                    mlm_model_type='roberta'
                )
            >>> nodes = n.get_nodes()
            >>> print(nodes)
            {
                'target_node': 'sentence',
                'similar_nodes': [[], [], []] # each list contains the similar nodes of the target word at a specific level
                'context_nodes': [[], [], []]
            }
        
        """
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


                for word in nodes['context_nodes'][level-1]:
                    similar_nodes += self._get_similar_nodes(word)
                    context_nodes += self._get_context_nodes(word)
                
                nodes['similar_nodes'].append(similar_nodes)
                nodes['context_nodes'].append(context_nodes)          
        return nodes
    
    def node_features(self, nodes: Dict[str, List[str]]) -> List[Dict[str, Union[str, int, List[float]]]]:
        node_features = []
        for level in range(self.level):
            for node_type in ['similar_nodes', 'context_nodes']:
                for node in nodes[node_type][level]:
                    node_feature = {
                        'node': node,
                        'node_type': node_type,
                        'node_level': level,
                        'frequency': sum(' ' + node + ' ' in s for s in self.dataset),
                        'embeddings': self.mlm.get_embedding(word=node)
                    }
                    node_features.append(node_feature)
        return node_features




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



if __name__ == '__main__':
    # data = ['this is a sentence', 'this is another sentence']
    # model_dir = 'output'

    # word2vec = Word2VecInference(f'{model_dir}/word2vec_aligned/word2vec_1980_aligned.model')
    # mlm = RobertaInference(f'{model_dir}/MLM_roberta_1980')
    # n = Nodes(
    #     target_word='sentence',
    #     dataset=data,
    #     level=3,
    #     k=2,
    #     c=2,
    #     word2vec_model = word2vec,
    #     mlm_model = mlm
    # )

    # nodes = n.get_nodes()
    # node_features = n.node_features(nodes)


