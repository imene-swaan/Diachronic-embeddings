from semantics.feature_extraction.roberta import RobertaInference
from semantics.feature_extraction.bert import BertInference
from semantics.feature_extraction.word2vec import Word2VecInference
from typing import List, Union, Dict
import torch
import numpy as np

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
            word2vec_model (Word2VecInference): the word2vec model's Inference class
            mlm_model (RobertaInference, BertInference): the MLM model's Inference class
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
        """
        nodes = {'target_node': [], 'similar_nodes': [], 'context_nodes': []}
        for level in range(self.level):
            if level == 0:
                similar_nodes = self._get_similar_nodes(self.target_word)
                context_nodes = self._get_context_nodes(self.target_word)

                nodes['similar_nodes'].append(similar_nodes)
                nodes['context_nodes'].append(context_nodes)
                nodes['target_node'].append([self.target_word])

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
    
    def get_node_features(self, nodes: Dict[str, List[str]]):
        """
        This method is used to get the features of the nodes of the word graph.

        Args:
            nodes (Dict[str, List[str]]): the nodes of the word graph

        Returns:
            - words (List[str]): the words of the nodes
            - node_ids (List[int]): the ids of the nodes. The target node has id 0.
            - node_features (np.ndarray): the features of the nodes of the word graph of shape (num_nodes, 3) where num_nodes is the number of nodes in the graph. The features are:
                - node_type: the type of the node (target: 0, similar: 1, context: 2).
                - node_level: the level of the node in the graph. The target node is level 0.
                - frequency: the frequency of the word node in the dataset.
            - embeddings (np.ndarray): the embeddings of the nodes of the word graph from the MLM model, of shape (num_nodes, 768).

        Examples:
            >>> word2vec = Word2VecInference('word2vec.model')
            >>> mlm = RobertaInference('MLM_roberta')
            >>> nodes = Nodes(target_word='sentence', dataset=['this is a sentence', 'this is another sentence'], level=3, k=2, c=2, word2vec_model = word2vec, mlm_model = mlm).get_nodes()
            >>> words, node_ids, node_features, embeddings = n.get_node_features(nodes)
            >>> print(words)
            ['sentence', 'this', 'is', 'a', 'another']
            >>> print(node_ids)
            [0, 1, 2, 3, 4]
            >>> print(node_features)
            [[0, 0, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2], [2, 1, 2]]
            >>> print(embeddings.shape)
            (5, 768)
        """
        words = []
        node_ids = []
        node_types = []
        node_levels = []
        frequencies = []
        embeddings = []
        count = 0
        for node_type in ['target_node', 'similar_nodes', 'context_nodes']:
            for level in range(len(nodes[node_type])):
                for node in nodes[node_type][level]:
                    words.append(node)
                    node_ids.append(count)
                    count += 1 
                    if node_type == 'target_node':
                        node_types.append(0)
                    elif node_type == 'similar_nodes':
                        node_types.append(1)
                    else:
                        node_types.append(2)
                    node_levels.append(level)
                    frequencies.append(sum(node in s for s in self.dataset))
                    embeddings.append(self.mlm.get_embedding(word=node).mean(axis=0))

        embeddings = np.array(embeddings)
        node_features = np.stack([node_types, node_levels, frequencies]).T
        # node_features = np.concatenate((node_features, embeddings), axis=1)
        return words, node_ids, node_features, embeddings



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



class Edges:
    def __init__(
            self,
            node_features: np.ndarray,
            node_embeddings: np.ndarray,
        ):

        self.node_features = node_features
        self.node_embeddings = node_embeddings


    def get_similarity(self, emb1: np.ndarray , emb2: np.ndarray) -> float:
        """
        This method is used to get the similarity between two nodes.

        Args:
            emb1 (np.ndarray): the embedding of the first word node
            emb2 (np.ndarray): the embedding of the second word node

        Returns:
            similarity (float): the similarity between the two embeddings
        """
        # np.dot(node1, node2) / (np.linalg.norm(node1) * np.linalg.norm(node2))
        return torch.cosine_similarity(torch.tensor(emb1).reshape(1,-1), torch.tensor(emb2).reshape(1,-1)).item()
    
    
    def get_pmi(self, node1: str, node2: str) -> float:
        """
        This method is used to get the PMI between two nodes.

        Args:
            node1 (str): the first node
            node2 (str): the second node

        Returns:
            pmi (float): the PMI between the two nodes
        """
        
        return 0.0





if __name__ == '__main__':
    data = ['this is a sentence', 'this is another sentence']
    model_dir = 'output'

    word2vec = Word2VecInference(f'{model_dir}/word2vec_aligned/word2vec_1980_aligned.model')
    mlm = RobertaInference(f'{model_dir}/MLM_roberta_1980')
    n = Nodes(
        target_word='sentence',
        dataset=data,
        level=3,
        k=2,
        c=2,
        word2vec_model = word2vec,
        mlm_model = mlm
    )

    nodes = n.get_nodes()
    words, ids, features, embeddings = n.get_node_features(nodes)

    e = Edges(
        node_features=features,
        node_embeddings=embeddings
    )
    
    sim = e.get_similarity(embeddings[0], embeddings[1])
    print(sim)


