from semantics.feature_extraction.roberta import RobertaInference
from semantics.feature_extraction.bert import BertInference
from semantics.feature_extraction.word2vec import Word2VecInference
from typing import List, Union, Dict, Optional, Tuple
import torch
import numpy as np
from math import log
from semantics.utils.utils import count_occurence, most_frequent
import tqdm
from semantics.utils.components import WordGraph

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
        __init__(self, target_word: str, dataset: List[str], level: int, k: int, c: int, word2vec_model_path: str, mlm_model_path: str, mlm_model_type: str = 'roberta') -> None
            The constructor of the Nodes class.
        get_similar_nodes(self, word: str) -> List[str]
            This method is used to get the similar nodes of a word.
        get_context_nodes(self, word: str) -> List[str]
            This method is used to get the context nodes of a word.
        get_nodes(self) -> Dict[str, List[str]]
            This method is used to get the nodes of the word graph.
        get_node_features(self, nodes: Dict[str, List[str]])
            This method is used to get the features of the nodes of the word graph.
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
    

    def get_similar_nodes(
            self, 
            word: Union[str, List[str]],
            keep_k: int = 50
            ) -> Dict[str, List[str]]:
        """
        This method is used to get the similar nodes of a word using the MLM model.
        
        Args:
            word (Union[str, List[str]]): the word to get the similar nodes for
            keep_k (int): the number of similar nodes to keep. Default: 50.
            
        Returns:
            similar_nodes (Dict[str, List[str]]): the similar nodes of the word

        Examples:
            >>> word2vec = Word2VecInference('word2vec.model')
            >>> mlm = RobertaInference('MLM_roberta')
            >>> nodes = Nodes(target_word='sentence', dataset=['this is a sentence', 'this is another sentence'], level=3, k=2, c=2, word2vec_model = word2vec, mlm_model = mlm)
            >>> similar_nodes = nodes.get_similar_nodes('sentence')
            >>> print(similar_nodes)
            {'sentence': ['sentence', 'sentence'], 'this': ['this', 'this'], 'is': ['is', 'is'], 'a': ['a', 'a'], 'another': ['another', 'another']}
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

               

    def get_context_nodes(
            self, 
            word: Union[str, List[str]],
            keep_k: int = 50
            ) -> Dict[str, List[str]]:
        """
        This method is used to get the context nodes of a word using the word2vec model.

        Args:
            word (Union[str, List[str]]): the word to get the context nodes for. If a list of words is given, the context nodes of all the words in the list are returned.

        Returns:
            context_nodes (Dict[str, List[str]]): the context nodes of the word

        Examples:
            >>> word2vec = Word2VecInference('word2vec.model')
            >>> mlm = RobertaInference('MLM_roberta')
            >>> nodes = Nodes(target_word='sentence', dataset=['this is a sentence', 'this is another sentence'], level=3, k=2, c=2, word2vec_model = word2vec, mlm_model = mlm)
            >>> context_nodes = nodes.get_context_nodes('sentence')
            >>> print(context_nodes)
            {'sentence': ['this', 'is'], 'this': ['sentence', 'is'], 'is': ['this', 'sentence'], 'a': ['this', 'is'], 'another': ['this', 'is']}
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
    
    def get_nodes(self) -> Dict[str, Dict[str, List[str]]]:
        """
        This method is used to get the nodes of the word graph (similar nodes, context nodes, and target node).

        Returns:
            nodes (Dict[str, Dict[str, List[str]]]): the nodes of the word graph. The keys are 'similar_nodes' and 'context_nodes'. The values are dictionaries with words and their similar/context nodes. The depth of word graph is determined by the level argument in the constructor.

        Examples:
            >>> word2vec = Word2VecInference('word2vec.model')
            >>> mlm = RobertaInference('MLM_roberta')
            >>> nd = Nodes(target_word='sentence', dataset=['this is a sentence', 'this is another sentence'], level=3, k=2, c=2, word2vec_model = word2vec, mlm_model = mlm)
            >>> nodes = nd.get_nodes()
            >>> print(nodes)
            {'similar_nodes': 
                {'sentence': ['sentence', 'sentence'], 'this': ['this', 'this'], 'is': ['is', 'is'], 'a': ['a', 'a'], 'another': ['another', 'another']}, 
            'context_nodes': 
                {'sentence': ['this', 'is'], 'this': ['sentence', 'is'], 'is': ['this', 'sentence'], 'a': ['this', 'is'], 'another': ['this', 'is']}
            }
        """
        nodes = {}
        for level in range(self.level):
            # nodes[level] = {}
            print(f'Getting the nodes of level {level} ...')

            if level == 0:
                similar_nodes  = self.get_similar_nodes(self.target_word, keep_k= 4)
                context_nodes = self.get_context_nodes(self.target_word, keep_k= 2)

                # nodes[level]['similar_nodes'] = similar_nodes
                # nodes[level]['context_nodes'] = context_nodes

                nodes['similar_nodes'] = similar_nodes
                nodes['context_nodes'] = context_nodes

            else:
                # previous_nodes = [node for node_list in nodes[level-1]['similar_nodes'].values() for node in node_list] + [node for node_list in nodes[level-1]['context_nodes'].values() for node in node_list]

                previous_nodes = [node for node_list in nodes['similar_nodes'].values() for node in node_list if node not in nodes['similar_nodes'].keys()] + [node for node_list in nodes['context_nodes'].values() for node in node_list if node not in nodes['context_nodes'].keys()]
               
                previous_nodes = list(set(previous_nodes))

                similar_nodes = self.get_similar_nodes(previous_nodes, keep_k= 2)
                context_nodes = self.get_context_nodes(previous_nodes, keep_k= 1)

                # nodes[level]['similar_nodes'] = similar_nodes
                # nodes[level]['context_nodes'] = context_nodes

                nodes['similar_nodes'].update(similar_nodes)
                nodes['context_nodes'].update(context_nodes)
                             
        return nodes
    
    def get_node_features(self, nodes: Dict[str, Dict[str, List[str]]]):
        """
        This method is used to get the features of the nodes of the word graph.

        Args:
            nodes (Dict[int, Dict[str, Dict[str, List[str]]]]): the nodes of the word graph. The keys are the levels of the graph. The values are dictionaries with the keys 'similar_nodes' and 'context_nodes'. The values of these keys are dictionaries with the keys 'target_word' and 'similar_nodes'/'context_nodes'. The values of these keys are lists of similar/context nodes.
        
        Returns:
            index (Dict): the index of the nodes of the word graph. Contains key_to_index and index_to_key dictionaries.
            node_features (np.ndarray): the features of the nodes of the word graph of shape (num_nodes, 3) where num_nodes is the number of nodes in the graph. The features are:
                - node_type: the type of the node (target, similar, context)
                - node_level: the level of the node in the graph
                - frequency: the frequency of the word node in the dataset
            embeddings (np.ndarray): the embeddings of the nodes of the word graph from the MLM model, of shape (num_nodes, 768).

        Examples:
            >>> word2vec = Word2VecInference('word2vec.model')
            >>> mlm = RobertaInference('MLM_roberta')
            >>> nodes = Nodes(target_word='sentence', dataset=['this is a sentence', 'this is another sentence'], level=3, k=2, c=2, word2vec_model = word2vec, mlm_model = mlm)
            >>> nd, nd_s = nodes.get_nodes()
            >>> index, node_features, embeddings = nodes.get_node_features(nd)
            >>> print(index)
            {'index_to_key': {0: 'sentence', 1: 'this', 2: 'is', 3: 'a', 4: 'another'}, 'key_to_index': {'sentence': 0, 'this': 1, 'is': 2, 'a': 3, 'another': 4}
            >>> print(node_features)
            [[0, 0, 2, ...], [1, 1, 2, ...], [1, 1, 2, ...], [1, 1, 2, ...], [2, 1, 2, ...]]
            >>> print(embeddings.shape)
            (5, 768)
        """
        words = [self.target_word]
        node_types = [1]
        # node_levels = [0]
        all_words_count = count_occurence(self.dataset)
        frequencies = [count_occurence(self.dataset, self.target_word)/ all_words_count]
        embeddings = [self.mlm.get_embedding(main_word=self.target_word).mean(axis=0)]

        # for level in range(self.level):
        #     for node_type in ['similar_nodes', 'context_nodes']:
        #         for node_list in nodes[level][node_type].values():
        #             for node in node_list:
        #                 if node in words:
        #                     continue

        #                 words.append(node)
        #                 if node_type == 'similar_nodes':
        #                     node_types.append(1)
        #                 else:
        #                     node_types.append(2)

        #                 node_levels.append(level+1)
        #                 frequencies.append(count_occurence(self.dataset, node) / all_words_count)
        #                 embeddings.append(self.mlm.get_embedding(main_word=node).mean(axis=0))
        

        for node_type in ['similar_nodes', 'context_nodes']:
            for node_list in nodes[node_type].values():
                for node in node_list:
                    if node in words:
                        continue

                    words.append(node)
                    if node_type == 'similar_nodes':
                        node_types.append(2)
                    else:
                        node_types.append(3)

                    frequencies.append(count_occurence(self.dataset, node) / all_words_count)
                    embeddings.append(self.mlm.get_embedding(main_word=node).mean(axis=0))


        index_to_key = {idx: word for idx, word in enumerate(words)}
        key_to_index = {word: idx for idx, word in enumerate(words)}  

        del words
        # print('Key to index: ', key_to_index)
        embeddings = np.array(embeddings)
        # node_features = np.stack([node_types, node_levels, frequencies]).T

        node_features = np.stack([node_types, frequencies]).T
        # node_features = np.concatenate((node_features, embeddings), axis=1)

        index = {'index_to_key': index_to_key, 'key_to_index': key_to_index}
        return index, node_features, embeddings



# edge_index:
# - target node -> similar node
# - target node -> context node
# - similar node -> similar node
# - similar node -> context node
# - context node -> context node

# edge features:
# - edge_type: the type of the edge (target-similar (1), target-context(2), similar-similar(3), similar-context(4), context-context(5), self-loop(0))
# - similarity: the similarity between node embeddings in the current snapshot
# - PMI: the PMI between nodes in the current snapshot

# - labels: 
    # - similarity: the similarity between similar nodes in the next snapshot



class Edges:
    """
    This class is used to get the edges of the word graph.

    methods:
        __init__(self, word_ids: Dict[int, str], node_features: np.ndarray, node_embeddings: np.ndarray)
            The constructor of the Edges class.
        get_similarity(self, emb1: int, emb2: int) -> float
            This method is used to get the similarity between two nodes.
        get_pmi(self, data: List[str], word1: str, word2: str) -> float
            This method is used to get the PMI between two nodes.
        get_edge_features(self, dataset: List[str])
            This method is used to get the edge features of the word graph.
    """
    def __init__(
            self,
            index: Dict,
            nodes: Dict[str, Dict[str, List[str]]],
            node_embeddings: np.ndarray,
        ):
        """
        Args:
            index (Dict): the index of the nodes of the word graph.
            nodes (Dict): the nodes of the word graph.
            node_strengths (Dict): the strength of the nodes of the word graph.
            node_embeddings (np.ndarray): the embeddings of the nodes of the word graph from the MLM model, of shape (num_nodes, 768).
        """

        self.index = index
        self.nodes = nodes
        self.node_embeddings = node_embeddings
       

    def get_similarity(self, emb1: int, emb2: int) -> float:
        """
        This method is used to get the similarity between two nodes.

        Args:
            emb1 (int): the first index of the embedding in node_embeddings
            emb2 (int): the second index of the embedding in node_embeddings

        Returns:
            similarity (float): the similarity between the two embeddings
        """
        # np.dot(node1, node2) / (np.linalg.norm(node1) * np.linalg.norm(node2))
        return torch.cosine_similarity(
            torch.tensor(self.node_embeddings[emb1]).reshape(1,-1), 
            torch.tensor(self.node_embeddings[emb2]).reshape(1,-1)
            ).item()
    
    
    def get_pmi(self, data: List[str], word1: str, word2: str) -> float:
        """
        This method is used to get the PMI between two nodes.

        Args:
            word1 (str): the first node (word)
            word2 (str): the second node (word)

        Returns:
            pmi (float): the PMI between the two words in the dataset
        """
        # Replace these methods with actual methods to get the word count, 
        # the co-occurrence count, and the total count.
        word1_count = count_occurence(data, word1)
        word2_count = count_occurence(data, word2)
        co_occurrence_count = count_occurence(data, [word1, word2])
        total_count = count_occurence(data)

        # Calculate probabilities
        p_word1 = word1_count / total_count
        p_word2 = word2_count / total_count
        p_co_occurrence = co_occurrence_count / total_count

        # Calculate PMI
        if p_co_occurrence > 0:
            pmi = log(p_co_occurrence / (p_word1 * p_word2), 2)
            npmi = pmi / (-log(p_co_occurrence, 2))
        
        else:
            npmi = 0
        
        return npmi
    
    def get_edge_features(self, dataset: List[str], sim_threshold: float = 0.5):
        """
        This method is used to get the edge features of the word graph.

        Args:
            dataset (List[str]): the dataset to get the edge features from
            sim_threshold (float): the similarity threshold to create an edge between two nodes. Default: 0.5.

        Returns:
            edge_index (np.ndarray): the edge index of the word graph of shape (2, num_edges) where num_edges is the number of edges in the graph. The first row contains the indices of the first node of the edge and the second row contains the indices of the second node of the edge. An edge is created if the similarity between the two nodes is greater than sim_threshold.
            edge_features (np.ndarray): the edge features of the word graph of shape (num_edges, 3) where num_edges is the number of edges in the graph. The features are:

                - edge_type: the type of the edge

                - similarity: the similarity between node embeddings in the current snapshot
                
                - PMI: the PMI between nodes in the current snapshot

        """
        edge_index_1 = []
        edge_index_2 = []
        edge_types = []
        similarities = []
        pmis = []
        edges = []
        # levels = max(self.nodes.keys()) + 1

        """
        Examples:
            >>> nodes = nd.get_nodes()
            >>> print(nodes)

            {'similar_nodes': 
                {
                    'trump': ['hand', 'deal'], 
                    'hand': ['play', 'win'], 
                    'deal': ['play', 'game'],
                    'bronzecolored: ['hand', 'deal'],
                }, 
            'context_nodes': 
                {
                    'trump': ['bronzecolored'],
                    'hand': ['shook'],
                    'deal': ['diamond'],
                    'bronzecolored: ['trump'],
                }
            }
        """

        for node_type in ['similar_nodes', 'context_nodes']:
            for source_node in self.nodes[node_type].keys():
                for target_node in self.nodes[node_type][source_node]:

                    source_idx = self.index['key_to_index'][source_node]
                    target_idx = self.index['key_to_index'][target_node]

                    # TODO: check if this is needed
                    if ((source_idx, target_idx) in edges) or ((target_idx, source_idx) in edges):
                        continue

                    similarity = self.get_similarity(source_idx, target_idx)

                    if similarity > sim_threshold:
                        edge_index_1.append(source_idx)
                        edge_index_2.append(target_idx)
                        edges.append((source_idx, target_idx))
                        edge_type = 0 if source_idx == target_idx else 1 if node_type == 'similar_nodes' else 2
                        edge_types.append(edge_type)
                        similarities.append(similarity)
                        pmi = self.get_pmi(dataset, source_node, target_node)
                        pmis.append(pmi)


        # for level in range(levels):
        #     for node_type in ['similar_nodes', 'context_nodes']:
        #         for node_1 in self.nodes[level][node_type].keys():
        #             for i, node_2 in enumerate(self.nodes[level][node_type][node_1]):

        #                 e1 = self.index['key_to_index'][node_1]
        #                 e2 = self.index['key_to_index'][node_2]


        #                 if ((e1,e2) in edges) or ((e2,e1) in edges):
        #                     continue

        #                 similarity = self.get_similarity(e1, e2)

        #                 if similarity > sim_threshold:
        #                     edge_index_1.append(e1)
        #                     edge_index_2.append(e2)
        #                     edges.append((e1,e2))
        #                     edge_type = 0 if e1 == e2 else 1 if node_type == 'similar_nodes' else 2
        #                     edge_types.append(edge_type)
        #                     similarities.append(similarity)
        #                     pmi = self.get_pmi(dataset, node_1, node_2)
        #                     pmis.append(pmi)

        del edges
        edge_index = np.stack([edge_index_1, edge_index_2])
        edge_features = np.stack([edge_types, similarities, pmis]).T
        return edge_index, edge_features




class TemporalGraph:
    """
    This class is used to get the temporal graph of a word.

    methods:
        __init__(self) -> None
            The constructor of the TemporalGraph class.
        __getitem__(self, idx) -> WordGraph
            Retrieves the snapshot at the specified index.
        add_graph(self, target_word: str, level: int, k: int, c: int, dataset: List[str], word2vec_model: Word2VecInference, mlm_model: Union[RobertaInference, BertInference])
            This method is used to add a snapshot to the temporal graph.
        construct_graph(self, current_index, current_node_feature_matrix, current_embeddings, current_edge_index, current_edge_feature_matrix)
            This method is used to construct the temporal graph.
        get_aligned_graph(self, current_graph: dict, previous_graph: dict) -> (dict, dict)
            This method is used to align the nodes of the current snapshot with the nodes of the previous snapshot.
        label_previous_graph(self, current_graph: dict, previous_graph: dict, label_feature_idx: int = 1) -> (np.ndarray, np.ndarray)
            This method is used to label the edges of the previous snapshot with the edge feature values in the current snapshot.
    """
    def __init__(
            self,
            index: Optional[List[dict]] = [],
            xs: Optional[List[np.ndarray]] = [],
            edge_indices: Optional[List[np.ndarray]] = [],
            edge_features: Optional[List[np.ndarray]] = [],
            ys: Optional[List[np.ndarray]] = [],
            y_indices: Optional[List[np.ndarray]] = []
            ) -> None:
        
        """
        Attributes:
            index (dict): the word index of the temporal graph. Contains key_to_index and index_to_key dictionaries.
            xs (List[np.ndarray]): the features of the nodes of the temporal graph.
            edge_indices (List[np.ndarray]): the edge index of the temporal graph.
            edge_features (List[np.ndarray]): the edge features of the temporal graph.
            ys (List[np.ndarray]): the labels of the edges of the temporal graph.
            y_indices (List[np.ndarray]): the indices of the labels of the edges of the temporal graph.
        
        """
        
        self.index =  index
        self.xs = xs
        self.edge_indices = edge_indices
        self.edge_features = edge_features
        self.ys = ys
        self.y_indices = y_indices

        self.nodes = []
    
    def __len__(self) -> int:
        """
        Returns the number of snapshots in the temporal graph.
        """
        return len(self.xs)

    def __getitem__(self, idx) -> WordGraph:
        """
        Retrieves the snapshot at the specified index.

        Parameters:
            idx (int): Index of the item to retrieve.

        Returns:
            graph (WordGraph): The snapshot at the specified index.
        """
        graph = WordGraph(
            index=self.index[idx],
            node_features= np.array(self.xs[idx]),
            edge_index= np.array(self.edge_indices[idx]),
            edge_features= np.array(self.edge_features[idx]),
            labels= np.array(self.ys[idx]),
            label_mask= np.array(self.y_indices[idx])
            )
        return graph

    def __setitem__(self, idx, key, value) -> None:
        """
        Sets the attribute at the specified index.

        Args:
            idx (int): Index of the item to set.
            key (str): Key of the attribute to set.
            value (any): Value to be set at the given index of the attribute.
        """
        valid_keys = ['index', 'xs', 'edge_indices', 'edge_features', 'ys', 'y_indices']
        if key in valid_keys:
            getattr(self, key)[idx] = value
        else:
            raise KeyError(f'Invalid key: {key}')
        

    def add_graph(
            self,
            target_word: str, 
            level: int, 
            k: int, 
            c: int,
            dataset: List[str], 
            word2vec_model: Word2VecInference, 
            mlm_model: Union[RobertaInference, BertInference],
            edge_threshold: float = 0.5,
            accumulate: bool = False
            ) -> Dict[str, Dict[str, List[str]]]:
        """
        This method is used to add a snapshot to the temporal graph.

        Args:
            target_word (str): the word to get the nodes for
            level (int): the level of the graph to get
            k (int): the number of similar nodes to get for each occurrence of the target word
            c (int): the number of context nodes to get for the target word
            dataset (List[str]): the sentences to get the nodes from
            word2vec_model (Word2VecInference): the word2vec model's Inference class
            mlm_model (RobertaInference, BertInference): the MLM model's Inference class
        
        Examples:
            >>> word2vec = Word2VecInference('word2vec.model')
            >>> mlm = RobertaInference('MLM_roberta')
            >>> tg = TemporalGraph()
            >>> tg.add_graph(target_word='sentence', level=3, k=2, c=2, dataset=['this is a sentence', 'this is another sentence'], word2vec_model = word2vec, mlm_model = mlm)
            >>> snapshot, node_features, edge_index, edge_feature, _, _= tg[0]
            >>> print(snapshot)
            {'index_to_key': {0: 'sentence', 1: 'this', 2: 'is', 3: 'a', 4: 'another'}, 'key_to_index': {'sentence': 0, 'this': 1, 'is': 2, 'a': 3, 'another': 4}
            >>> print(node_features)
            [[0, 0, 2, ...], [1, 1, 2, ...], [1, 1, 2, ...], [1, 1, 2, ...], [2, 1, 2, ...]]
            >>> print(edge_index)
            [[0, 0, 0, 1, 1, 2, 2, 3, 3, 4], [1, 2, 4, 1, 3, 1, 4, 1, 4, 4]]
            >>> print(edge_feature)
            [[0, 0.9999999403953552, 0.0], [0, 0.9999999403953552, 0.0], [0, 0.9999999403953552, 0.0], [0, 0.9999999403953552, 0.0], [0, 0.9999999403953552, 0.0], [1, 0.9999999403953552, 0.0], [1, 0.9999999403953552, 0.0], [1, 0.9999999403953552, 0.0], [1, 0.9999999403953552, 0.0], [1, 0.9999999403953552, 0.0]]

            >>> tg.add_graph(target_word='sentence', level=3, k=2, c=2, dataset=['this is a sentence', 'this is another sentence', 'this is a third sentence'], word2vec_model = word2vec, mlm_model = mlm)
            >>> _, _, _, _, labels, label_mask= tg[0]
            >>> print(labels)
            [[0.9999, 1., 0.0001]]
            >>> print(label_mask)
            [[0, 1, 0], [1, 2, 4]]
        """

        print(f'Adding the nodes of the word graph for the word "{target_word}"...')
        nd = Nodes(
            target_word= target_word,
            dataset=dataset,
            level= level,
            k= k,
            c= c,
            word2vec_model = word2vec_model,
            mlm_model = mlm_model
            )

        nodes = nd.get_nodes()

        if accumulate and (len(self.nodes) > 0):
            print('Accumulating the nodes of the word graph...')
            previous_nodes = self.nodes[-1]
            for node_type in ['similar_nodes', 'context_nodes']:
                for node in previous_nodes[node_type].keys():
                    if node not in nodes[node_type].keys():
                        nodes[node_type][node] = previous_nodes[node_type][node]
                    else:
                        nodes[node_type][node] += previous_nodes[node_type][node]
                        nodes[node_type][node] = list(set(nodes[node_type][node]))
                        
    

        print('Getting their features...', '\n')
        index, node_feature_matrix, embeddings = nd.get_node_features(nodes)
        print(f'Adding the edges of the word graph for the word "{target_word}"...')
        edges = Edges(
            index=index,
            nodes=nodes,
            node_embeddings=embeddings
        )
        edge_index, edge_feature_matrix = edges.get_edge_features(dataset, sim_threshold=edge_threshold)

        print('Constructing the temporal graph...', '\n')

        self.index.append(index)
        self.xs.append(np.concatenate((node_feature_matrix, embeddings), axis=1))
        self.edge_indices.append(edge_index)
        self.edge_features.append(edge_feature_matrix)
        self.ys.append(np.array([]))
        self.y_indices.append(np.array([]))

        self.nodes.append(nodes)
        return nodes
    
    def align_graphs(self) -> None:
        all_words = [set(self[i].index['key_to_index'].keys()) for i in range(len(self.index))]
        dynamic_graph = is_dynamic(all_words)

        if not dynamic_graph:
            print('The graph nodes are static...', '\n')
            reference_index = self[0].index

            for i in range(1, len(self.index)):
                next_index = self[i].index
                index_mapping = {next_index['key_to_index'][key]: reference_index['key_to_index'][key] for key in next_index['key_to_index'].keys()}

                reordered_node_feature_matrix = np.zeros_like(self[i].node_features)
                for next_idx, ref_idx in index_mapping.items():
                    reordered_node_feature_matrix[ref_idx] = self[i].node_features[next_idx]

                updated_edge_index = np.zeros_like(self[i].edge_index)
                for j in range(self[i].edge_index.shape[1]):
                    updated_edge_index[0, j] = index_mapping[self[i].edge_index[0, j]]
                    updated_edge_index[1, j] = index_mapping[self[i].edge_index[1, j]]


                self.index[i] = reference_index
                self.xs[i] = reordered_node_feature_matrix
                self.edge_indices[i] = updated_edge_index


        else:
            print('The graph nodes are dynamic...', '\n')
            all_words = set().union(*all_words)
            unified_dict = {word: idx for idx, word in enumerate(all_words)}
            unified_dict_reverse = {idx: word for idx, word in enumerate(all_words)}
            reordered_index = {'index_to_key': unified_dict_reverse, 'key_to_index': unified_dict}

            for i in range(len(self.index)):
                snap_index = self[i].index
                index_mapping = {snap_index['key_to_index'][key]: unified_dict[key] for key in snap_index['key_to_index'].keys()}

                reordered_node_feature_matrix = np.zeros((len(unified_dict), self[i].node_features.shape[1]))
                for snap_idx, unified_idx in index_mapping.items():
                    reordered_node_feature_matrix[unified_idx] = self[i].node_features[snap_idx]
                
                updated_previous_edge_index = np.zeros(self[i].edge_index.shape)
                for e in range(self[i].edge_index.shape[1]):
                    n1 = self[i].edge_index[0, e]
                    n2 = self[i].edge_index[1, e]

                    updated_previous_edge_index[0, e] = index_mapping[n1]
                    updated_previous_edge_index[1, e] = index_mapping[n2]
                
                self.index[i] = reordered_index
                self.xs[i] = reordered_node_feature_matrix
                self.edge_indices[i] = updated_previous_edge_index
    
    def label_graphs(self, label_feature_idx: int = 1) -> None:
        """
        This method is used to label the edges of the temporal graph with the edge feature values in the next snapshot.

        Args:
            label_feature_idx (int): the index of the edge feature to use as a label. Default: 1.
        """
        for i in range(len(self.xs)-1):
            current_graph = self[i]
            next_graph = self[i+1]

            current_edge_index = current_graph.edge_index

            next_edge_index = next_graph.edge_index
            next_edge_features = next_graph.edge_features

            current_edges = [tuple(edge) for edge in current_edge_index.T]
            next_edges  = [tuple(edge) for edge in next_edge_index.T]

            labels = []
            label_mask_1 = []
            label_mask_2 = []

            for edge in current_edges:
                if edge in next_edges:
                    label_mask_1.append(edge[0])
                    label_mask_2.append(edge[1])

                    next_index = next_edges.index(edge)
                    label = next_edge_features[next_index][label_feature_idx]
                    labels.append(label)
            
            
            self.ys[i] = np.array(labels)
            self.y_indices[i] = np.stack([label_mask_1, label_mask_2])
            

def is_dynamic(sets):
    union_of_all_sets = set().union(*sets)
    return not all(s == union_of_all_sets for s in sets)

















        # self.construct_graph(
        #     current_index=index,
        #     current_node_feature_matrix=node_feature_matrix,
        #     current_embeddings=embeddings,
        #     current_edge_index=edge_index,
        #     current_edge_feature_matrix=edge_feature_matrix
        # )

        # return nd, nd_s

    # def construct_graph(
    #         self, 
    #         current_index, 
    #         current_node_feature_matrix, 
    #         current_embeddings, 
    #         current_edge_index, 
    #         current_edge_feature_matrix
    #         ):
        
    #     """
    #     This method is used to construct the temporal graph.

    #     Args:
    #         current_index (dict): the index of the nodes of the current snapshot.
    #         current_node_feature_matrix (np.ndarray): the features of the nodes of the current snapshot.
    #         current_embeddings (np.ndarray): the embeddings of the nodes of the current snapshot.
    #         current_edge_index (np.ndarray): the edge index of the current snapshot.
    #         current_edge_feature_matrix (np.ndarray): the edge features of the current snapshot.
    #     """
        
    #     if len(self.xs) == 0:
    #         print('Adding the first snapshot to the temporal graph...', '\n')
    #         self.index = current_index
    #         self.xs.append(np.concatenate((current_node_feature_matrix, current_embeddings), axis=1))
    #         self.edge_indices.append(current_edge_index)
    #         print('\ncurrent_edge_index: ', current_edge_index.shape)
    #         self.edge_features.append(current_edge_feature_matrix)
    #         print('current_edge_feature_matrix: ', current_edge_feature_matrix.shape, '\n')
    #         self.ys.append(np.array([]))
    #         self.y_indices.append(np.array([]))

    #     else:
    #         print(f'Adding the {len(self.xs) + 1} snapshot to the temporal graph...', '\n')

    #         previous_graph = {
    #             'index': self.index,
    #             'node_features': self[-1].node_features,
    #             'edge_index': self[-1].edge_index,
    #             'edge_features': self[-1].edge_features
    #         }

    #         current_graph = {
    #             'index': current_index,
    #             'node_features': np.concatenate((current_node_feature_matrix, current_embeddings), axis=1),
    #             'edge_index': current_edge_index,
    #             'edge_features': current_edge_feature_matrix
    #         }


    #         aligned_previous_graph, aligned_current_graph = self.get_aligned_graph(current_graph, previous_graph)

    #         print('\ncurrent_edge_index: ', aligned_current_graph['edge_index'].shape)
    #         print('current_edge_feature_matrix: ', aligned_current_graph['edge_features'].shape, '\n')

    #         print('\nprevious_edge_index: ', aligned_previous_graph['edge_index'].shape)
    #         print('previous_edge_feature_matrix: ', aligned_previous_graph['edge_features'].shape, '\n')


    #         print('Labeling the edges...', '\n')
    #         previous_labels, previous_label_mask = self.label_previous_graph(current_graph, previous_graph)

    #         print('Previous labels: ', previous_labels.shape)
    #         print('Previous label mask: ', previous_label_mask.shape, '\n')

    #         self.index = aligned_previous_graph['index']

    #         self.xs[-1] = aligned_previous_graph['node_features']
    #         self.edge_indices[-1] = aligned_previous_graph['edge_index']
    #         self.edge_features[-1] = aligned_previous_graph['edge_features']
    #         self.ys[-1] = previous_labels
    #         self.y_indices[-1] = previous_label_mask

            
    #         self.xs.append(aligned_current_graph['node_features'])
    #         self.edge_indices.append(aligned_current_graph['edge_index'])
    #         self.edge_features.append(aligned_current_graph['edge_features'])
    #         self.ys.append(np.array([]))
    #         self.y_indices.append(np.array([]))

            
    
    # def get_aligned_graph(
    #         self, 
    #         current_graph: dict, 
    #         previous_graph: dict
    #         ) -> (dict, dict):
        
    #     """
    #     This method is used to align the nodes of the current snapshot with the nodes of the previous snapshot.

    #     Args:
    #         current_graph (dict): the current snapshot of the temporal graph to align with the previous snapshot.
    #         previous_graph (dict): the previous snapshot of the temporal graph to align with the current snapshot.

    #     Returns:
    #         aligned_previous_graph (dict): the aligned previous snapshot of the temporal graph.
    #         aligned_current_graph (dict): the aligned current snapshot of the temporal graph.
    #     """

    #     current_index = current_graph['index']
    #     previous_index = previous_graph['index']

    #     if current_index == previous_index:
    #         return current_graph

    #     current_words = set(current_index['key_to_index'].keys())
    #     previous_words = set(previous_index['key_to_index'].keys())
    
    #     dynamic_graph = current_words != previous_words

    #     if not dynamic_graph:
    #         print('The graph is static...', '\n')
    #         index_mapping = {current_index['key_to_index'][key]: previous_index['key_to_index'][key] for key in current_index['key_to_index'].keys()}

    #         reordered_node_feature_matrix = np.zeros_like(current_graph['node_features'])
    #         for current_idx, previous_idx in index_mapping.items():
    #             reordered_node_feature_matrix[previous_idx] = current_graph['node_features'][current_idx]


    #         updated_edge_index = np.zeros_like(current_graph['edge_index'])
    #         for i in range(current_graph['edge_index'].shape[1]):
    #             updated_edge_index[0, i] = index_mapping[current_graph['edge_index'][0, i]]
    #             updated_edge_index[1, i] = index_mapping[current_graph['edge_index'][1, i]]

    #         aligned_current_graph = {
    #             'index': previous_graph['index'],
    #             'node_features': reordered_node_feature_matrix,
    #             'edge_index': updated_edge_index,
    #             'edge_features': current_graph['edge_features']
    #         }
    #         return previous_graph, aligned_current_graph

        
    #     else:
    #         print('The graph is dynamic...', '\n')
    #         all_words = previous_words | current_words
    #         unified_dict = {word: idx for idx, word in enumerate(all_words)}
    #         unified_dict_reverse = {idx: word for idx, word in enumerate(all_words)}
    #         reordered_index = {'index_to_key': unified_dict_reverse, 'key_to_index': unified_dict}

    #         # print('Previous key to index: ', len(previous_index['key_to_index']), previous_index['key_to_index'], '\n')
    #         # print('Current key to index: ', len(current_index['key_to_index']), current_index['key_to_index'], '\n')
    #         # print('Unified index: ', len(reordered_index['key_to_index']), reordered_index['key_to_index'], '\n')

    #         # print('Number of previous words', sum([1 for _ in previous_index['key_to_index'].keys()]))
    #         # print('Number of current words', sum([1 for _ in current_index['key_to_index'].keys()]), '\n')
            
            
    #         reordered_previous_node_feature_matrix = np.zeros((len(unified_dict), previous_graph['node_features'].shape[1]))
    #         for word, index in previous_index['key_to_index'].items():
    #             if word in unified_dict:
    #                 reordered_previous_node_feature_matrix[unified_dict[word]] = previous_graph['node_features'][index]
            
    #         # first_word = list(previous_index['key_to_index'].keys())[0]
    #         # print('first word: ', first_word, '\n')
    #         # print('Index in previous graph of first word: ', previous_index['key_to_index'][first_word], '\n')
    #         # print('Index in current graph of first word: ', current_index['key_to_index'][first_word], '\n')
    #         # print('Index in aligned previous graph of first word: ', unified_dict[first_word], '\n')

    #         # print('Feature in previous graph of first word is equal to aligned one: ', np.array_equal(reordered_previous_node_feature_matrix[unified_dict[first_word]], previous_graph['node_features'][previous_index['key_to_index'][first_word]]), '\n')
    #         # print('Feature in current graph of first word is equal to aligned one: ', np.array_equal(reordered_previous_node_feature_matrix[unified_dict[first_word]], current_graph['node_features'][current_index['key_to_index'][first_word]]), '\n')


    #         reordered_current_node_feature_matrix = np.zeros((len(unified_dict), current_graph['node_features'].shape[1]))

    #         # print('Shape of previous node features: ', previous_graph['node_features'].shape, '\n')
    #         # print('New shape after alignement: ', reordered_previous_node_feature_matrix.shape, '\n')
    #         for word, index in current_index['key_to_index'].items():
    #             if word in unified_dict:
    #                 reordered_current_node_feature_matrix[unified_dict[word]] = current_graph['node_features'][index]

    
    #         # Mapping old indices to new indices for the previous dictionary
    #         previous_index_mapping = {old_index: unified_dict[word] for word, old_index in previous_index['key_to_index'].items()}

    #         # print('Previous key to index: ', previous_index['key_to_index'], '\n')
    #         # print('Previous to new index mapping: ', previous_index_mapping, '\n')
    #         # print('Edge index: ', previous_graph['edge_index'], '\n')

    #         updated_previous_edge_index = np.zeros(previous_graph['edge_index'].shape)
            
    #         # print('Previous edge index: ', previous_graph['edge_index'].shape, '\n')
    #         # print('Updated previous edge index: ', updated_previous_edge_index.shape, '\n')
    #         for i in range(previous_graph['edge_index'].shape[1]):
    #             n1 = previous_graph['edge_index'][0, i]
    #             n2 = previous_graph['edge_index'][1, i]
    #             # print('Old edge: ', (n1, n2))
    #             # print('New edge: ', (previous_index_mapping[n1], previous_index_mapping[n2]), '\n')
    #             updated_previous_edge_index[0, i] = previous_index_mapping[n1]
    #             updated_previous_edge_index[1, i] = previous_index_mapping[n2]
    #         # print('Updated edge index: ', updated_previous_edge_index.shape, '\n')

    #         # Mapping old indices to new indices for the current dictionary
    #         current_index_mapping = {old_index: unified_dict[word] for word, old_index in current_index['key_to_index'].items()}
    #         updated_current_edge_index = np.zeros(current_graph['edge_index'].shape)
    #         for i in range(current_graph['edge_index'].shape[1]):
    #             updated_current_edge_index[0, i] = current_index_mapping[current_graph['edge_index'][0, i]]
    #             updated_current_edge_index[1, i] = current_index_mapping[current_graph['edge_index'][1, i]]
            
            
    #         aligned_previous_graph = {
    #             'index': reordered_index,
    #             'node_features': reordered_previous_node_feature_matrix,
    #             'edge_index': updated_previous_edge_index,
    #             'edge_features': previous_graph['edge_features']
    #         }

    #         aligned_current_graph = {
    #             'index': reordered_index,
    #             'node_features': reordered_current_node_feature_matrix,
    #             'edge_index': updated_current_edge_index,
    #             'edge_features': current_graph['edge_features']
    #         }

    #         return aligned_previous_graph, aligned_current_graph
            
   

    # def label_previous_graph(
    #         self,
    #         current_graph: dict,
    #         previous_graph: dict,
    #         label_feature_idx: int = 1
    #         ) -> (np.ndarray, np.ndarray):
    #     """
    #     This method is used to label the edges of the previous snapshot with the edge feature values in the current snapshot.

    #     Args:
    #         current_graph (dict): the current snapshot of the temporal graph to use for labeling the previous snapshot.
    #         previous_graph (dict): the previous snapshot of the temporal graph to label.
    #         label_feature_idx (int): the index of the feature to use as labels. Default: 1.

    #     Returns:
    #         labels (np.ndarray): the labels of the edges of the graph at the specified index.
    #         labels_mask (np.ndarray): the indices of the labels of the edges of the graph at the specified index.
    #     """

    #     current_edge_index = current_graph['edge_index']
    #     current_edge_features = current_graph['edge_features']

    #     previous_edge_index = previous_graph['edge_index']

    #     previous_edges = [tuple(edge) for edge in previous_edge_index.T]
    #     current_edges  = [tuple(edge) for edge in current_edge_index.T]

    #     labels = []
        
    #     label_mask_1 = []
    #     label_mask_2 = []

    #     for _, previous_edge in enumerate(previous_edges):
    #         if previous_edge in current_edges:
    #             label_mask_1.append(previous_edge[0])
    #             label_mask_2.append(previous_edge[1])

    #             current_index = current_edges.index(previous_edge)
    #             labels.append(current_edge_features[current_index][label_feature_idx])

    #     label_mask = np.stack([label_mask_1, label_mask_2])
    #     labels = np.array(labels)

    #     return labels, label_mask





            








# if __name__ == '__main__':
    # data1 = ['this is a sentence', 'this is another sentence']
    # data2 = ['this is a sentence', 'this is another sentence', 'this is a third sentence']
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
    # words_ids, features, embeddings = n.get_node_features(nodes)

    # e = Edges(
    #     word_ids=words_ids,
    #     node_features=features,
    #     node_embeddings=embeddings
    # )
    
    # edge_index, edge_features = e.get_edge_features(dataset=data)

    # print('Words:', words_ids, '\n')
    # print('Edge index:', edge_index, '\n')
    # print('Edge features:', edge_features, '\n')
    # print('Shape edge_features: ', edge_features.shape)

    # tg = TemporalGraph()

    # tg.add_graph(
    #     target_word='sentence',
    #     level=3,
    #     k=2,
    #     c=2,
    #     dataset=data1,
    #     word2vec_model=word2vec,
    #     mlm_model=mlm
    # )

    # tg.add_graph(
    #     target_word='sentence',
    #     level=3,
    #     k=2,
    #     c=2,
    #     dataset=data2,
    #     word2vec_model=word2vec,
    #     mlm_model=mlm
    # )
    
    # print('First snapshot:', tg[0], '\n')
    # print('Second snapshot:', tg[1], '\n')
    