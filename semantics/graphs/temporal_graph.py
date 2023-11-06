from semantics.feature_extraction.roberta import RobertaInference
from semantics.feature_extraction.bert import BertInference
from semantics.feature_extraction.word2vec import Word2VecInference
from typing import List, Union, Dict
import torch
import numpy as np
from math import log
from semantics.utils.utils import count_occurence

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
            - index (Dict[str, Dict[int, str]]): the index of the nodes of the word graph. The keys are 'index_to_key' and 'key_to_index'. The values are dictionaries where the keys are the indices of the nodes and the values are the words of the nodes.
            - node_features (np.ndarray): the features of the nodes of the word graph of shape (num_nodes, 3) where num_nodes is the number of nodes in the graph. The features are:
                - node_type: the type of the node (target: 0, similar: 1, context: 2).
                - node_level: the level of the node in the graph. The target node is level 0.
                - frequency: the frequency of the word node in the dataset.
            - embeddings (np.ndarray): the embeddings of the nodes of the word graph from the MLM model, of shape (num_nodes, 768).

        Examples:
            >>> word2vec = Word2VecInference('word2vec.model')
            >>> mlm = RobertaInference('MLM_roberta')
            >>> nodes = Nodes(target_word='sentence', dataset=['this is a sentence', 'this is another sentence'], level=3, k=2, c=2, word2vec_model = word2vec, mlm_model = mlm).get_nodes()
            >>> index, node_features, embeddings = n.get_node_features(nodes)
            >>> print(index)
            {'index_to_key': {0: 'sentence', 1: 'this', 2: 'is', 3: 'a', 4: 'another'}, 'key_to_index': {'sentence': 0, 'this': 1, 'is': 2, 'a': 3, 'another': 4}
            >>> print(node_features)
            [[0, 0, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2], [2, 1, 2]]
            >>> print(embeddings.shape)
            (5, 768)
        """
        index_to_key = {}
        key_to_index = {}
        node_types = []
        node_levels = []
        frequencies = []
        embeddings = []
        count = 0
        for node_type in ['target_node', 'similar_nodes', 'context_nodes']:
            for level in range(len(nodes[node_type])):
                for node in nodes[node_type][level]:
                    index_to_key[count] = node
                    key_to_index[node] = count
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
            index_to_key: Dict[int, str],
            node_features: np.ndarray,
            node_embeddings: np.ndarray,
        ):

        self.index_to_key = index_to_key
        self.node_features = node_features
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
            word_idx1 (int): the index of the first node
            word_idx2 (int): the index of the second node

        Returns:
            pmi (float): the PMI between the two nodes
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
        pmi = log(p_co_occurrence / (p_word1 * p_word2), 2) if p_co_occurrence > 0 else 0
        return pmi
    
    def get_edge_features(self, dataset: List[str]):
        """
        This method is used to get the edge features of the word graph.

        Args:
            dataset (List[str]): the dataset to get the edge features from

        Returns:
            - edge_index (np.ndarray): the edge index of the word graph of shape (2, num_edges) where num_edges is the number of edges in the graph. The first row contains the indices of the first node of the edge and the second row contains the indices of the second node of the edge.
            - edge_features (np.ndarray): the edge features of the word graph of shape (num_edges, 3) where num_edges is the number of edges in the graph. The features are:
                - edge_type: the type of the edge (target-similar (1), target-context(2), similar-similar(3), similar-context(4), context-context(5), self-loop(0))
                - similarity: the similarity between node embeddings in the current snapshot
                - PMI: the PMI between nodes in the current snapshot

        """
        edge_index_1 = []
        edge_index_2 = []
        edge_types = []
        similarities = []
        pmis = []
        for word_idx1 in range(self.node_features.shape[0]):
            for word_idx2 in range(word_idx1, self.node_features.shape[0]):
                if word_idx1 == word_idx2:
                    edge_type = 0
                elif self.node_features[word_idx1][0] == 0 and self.node_features[word_idx2][0] == 1:
                    edge_type = 1
                elif self.node_features[word_idx1][0] == 0 and self.node_features[word_idx2][0] == 2:
                    edge_type = 2
                elif self.node_features[word_idx1][0] == 1 and self.node_features[word_idx2][0] == 1:
                    edge_type = 3
                elif self.node_features[word_idx1][0] == 1 and self.node_features[word_idx2][0] == 2:
                    edge_type = 4
                elif self.node_features[word_idx1][0] == 2 and self.node_features[word_idx2][0] == 2:
                    edge_type = 5

                similarity = self.get_similarity(word_idx1, word_idx2)
                pmi = self.get_pmi(dataset, self.index_to_key[word_idx1], self.index_to_key[word_idx2])

                if similarity > 0.5:
                    edge_index_1.append(word_idx1)
                    edge_index_2.append(word_idx2)
                    edge_types.append(edge_type)
                    similarities.append(similarity)
                    pmis.append(pmi)

        print('Similarities: ', similarities, '\n')
        print('PMIs: ', pmis)
        edge_index = np.stack([edge_index_1, edge_index_2])
        edge_features = np.stack([edge_types, similarities, pmis]).T

        return edge_index, edge_features




class TemporalGraph:
    def __init__(
            self
            ):

        self.snapshots = []
        self.xs = []
        self.edge_indices = []
        self.edge_features = []
        self.ys = []
        self.y_indices = []

    def __getitem__(self, idx):
        """
        Retrieves the snapshot at the specified index.

        Parameters:
            idx (int): Index of the item to retrieve.

        Returns:
            snapshot (dict): the graph data at the specified index.
        """
        # Get the tokenized inputs at the specified index
        snapshot = self.snapshots[idx]
        node_features = self.xs[idx]
        edge_index = self.edge_indices[idx]
        edge_feature = self.edge_features[idx]
        labels = self.ys[idx]
        labels_mask = self.y_indices[idx]

        return snapshot, node_features, edge_index, edge_feature, labels, labels_mask


    def add_snapshot(
            self,
            target_word: str, 
            level: int, 
            k: int, 
            c: int,
            dataset: List[str], 
            word2vec_model: Word2VecInference, 
            mlm_model: Union[RobertaInference, BertInference]
            ) -> None:
        
        nodes = Nodes(
            target_word= target_word,
            dataset=dataset,
            level= level,
            k= k,
            c= c,
            word2vec_model = word2vec_model,
            mlm_model = mlm_model
            ).get_nodes()
        
        index, node_feature_matrix, embbeddings = Nodes.get_node_features(nodes)

        edges = Edges(
            index_to_key=index['index_to_key'],
            node_features=node_feature_matrix,
            node_embeddings=embbeddings
        )
        edge_index, edge_feature_matrix = edges.get_edge_features(dataset)
        labels, labels_mask = self.label_snapshot(len(self.snapshots)-1)

        self.snapshots.append(index)
        self.xs.append(node_feature_matrix)
        self.edge_indices.append(edge_index)
        self.edge_features.append(edge_feature_matrix)
        self.ys.append(labels)
        self.y_indices.append(labels_mask)
        
    
    def label_snapshot(self, current_snapshot: int, label_feature_idx: int = 1) -> None:
        if current_snapshot >= len(self.snapshots) - 1:
            raise ValueError('The labels of the current snapshot do not exist.')
        
        current_snapshot_index = self.snapshots[current_snapshot]
        next_snapshot_index, _, next_edge_index, next_edge_features, _, _ = self.__getitem__(current_snapshot + 1)


        current_keys = set(current_snapshot_index['key_to_index'].keys())
        next_keys = set(next_snapshot_index['key_to_index'].keys())

        dynamic_graph = current_keys != next_keys

        if not dynamic_graph:
            self.ys[current_snapshot] = next_edge_features[:, label_feature_idx]
            self.y_indices[current_snapshot] = next_edge_index

        else:
            static_nodes = current_keys.intersection(next_keys)
            static_nodes_idx = [current_snapshot_index['key_to_index'][key] for key in static_nodes]

            dynamic_nodes = next_keys.difference(current_keys)
            dynamic_nodes_idx = [next_snapshot_index['key_to_index'][key] for key in dynamic_nodes]
            







            








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
    words_ids, features, embeddings = n.get_node_features(nodes)

    e = Edges(
        word_ids=words_ids,
        node_features=features,
        node_embeddings=embeddings
    )
    
    edge_index, edge_features = e.get_edge_features(dataset=data)

    # print('Words:', words_ids, '\n')
    # print('Edge index:', edge_index, '\n')
    # print('Edge features:', edge_features, '\n')
    # print('Shape edge_features: ', edge_features.shape)


