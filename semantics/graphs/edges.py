from typing import List, Tuple
import torch
import numpy as np
from math import log
from semantics.utils.utils import count_occurence
from semantics.utils.components import GraphNodes, GraphIndex



class Edges:
    """
    """
    def __init__(
            self,
            index: GraphIndex,
            nodes: GraphNodes,
            node_embeddings: np.ndarray,
        ):
        """
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
    
    def get_edge_features(self, dataset: List[str], sim_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        edge_index_1 = []
        edge_index_2 = []
        edge_types = []
        similarities = []
        pmis = []
        edges = []

        if self.nodes.similar_nodes is not None:
            for source_node in self.nodes.similar_nodes.keys():
                for target_node in self.nodes.similar_nodes[source_node]:
                    source_idx = self.index.key_to_index[source_node]
                    target_idx = self.index.key_to_index[target_node]

                    if ((source_idx, target_idx) in edges) or ((target_idx, source_idx) in edges):
                        continue

                    similarity = self.get_similarity(source_idx, target_idx)

                    if similarity > sim_threshold:
                        edge_index_1.append(source_idx)
                        edge_index_2.append(target_idx)
                        edges.append((source_idx, target_idx))
                        edge_types.append(1)
                        similarities.append(similarity)
                        pmis.append(self.get_pmi(dataset, source_node, target_node))

        if self.nodes.context_nodes is not None:
            for source_node in self.nodes.context_nodes.keys():
                for target_node in self.nodes.context_nodes[source_node]:

                    source_idx = self.index.key_to_index[source_node]
                    target_idx = self.index.key_to_index[target_node]

                    if ((source_idx, target_idx) in edges) or ((target_idx, source_idx) in edges):
                        continue

                    similarity = self.get_similarity(source_idx, target_idx)

                    if similarity > sim_threshold:
                        edge_index_1.append(source_idx)
                        edge_index_2.append(target_idx)
                        edges.append((source_idx, target_idx))
                        edge_types.append(2)
                        similarities.append(similarity)
                        pmis.append(self.get_pmi(dataset, source_node, target_node))

        del edges
        edge_index = np.stack([edge_index_1, edge_index_2])
        edge_features = np.stack([edge_types, similarities, pmis]).T

        assert edge_index.shape[1] == edge_features.shape[0]
        return edge_index, edge_features

