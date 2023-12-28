from semantics.feature_extraction.roberta import RobertaInference
from semantics.feature_extraction.bert import BertInference
from semantics.feature_extraction.word2vec import Word2VecInference
from typing import List, Union, Dict, Optional, Tuple
import numpy as np
from semantics.utils.components import WordGraph, GraphNodes, GraphIndex
from semantics.graphs.edges import Edges
from semantics.graphs.nodes import Nodes







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
            index: Optional[List[GraphIndex]] = [],
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

        self.nodes: List[GraphNodes] = []
        
    
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
   


    def add_graph(
            self,
            target_word: Union[str, List[str], Dict[str, List[str]]], 
            level: int, 
            k: int, 
            c: int,
            dataset: List[str], 
            word2vec_model: Word2VecInference, 
            mlm_model: Union[RobertaInference, BertInference],
            edge_threshold: float = 0.5,
            accumulate: bool = False,
            keep_k: Optional[Dict[int, Tuple[int, int]]] = None
            ) -> None:
        """
        """

        # print(f'Adding the nodes of the word graph for the word "{target_word}"...')
        nodes = Nodes(
            target= target_word,
            dataset=dataset,
            level= level,
            k= k,
            c= c,
            word2vec_model = word2vec_model,
            mlm_model = mlm_model,
            keep_k= keep_k
            )

        nodes.generate_nodes()

        
        if accumulate and (len(self.nodes) > 0):
            print('Accumulating the nodes of the word graph...')
            previous_nodes = self.nodes[-1]

            for similar_node in previous_nodes.similar_nodes.keys():
                if similar_node not in nodes.graph_nodes.similar_nodes.keys():
                    nodes.graph_nodes.similar_nodes[similar_node] = previous_nodes.similar_nodes[similar_node]
                else:
                    nodes.graph_nodes.similar_nodes[similar_node] += previous_nodes.similar_nodes[similar_node]
                    nodes.graph_nodes.similar_nodes[similar_node] = list(set(nodes.graph_nodes.similar_nodes[similar_node]))

            for context_node in previous_nodes.context_nodes.keys():
                if context_node not in nodes.graph_nodes.context_nodes.keys():
                    nodes.graph_nodes.context_nodes[context_node] = previous_nodes.context_nodes[context_node]
                else:
                    nodes.graph_nodes.context_nodes[context_node] += previous_nodes.context_nodes[context_node]
                    nodes.graph_nodes.context_nodes[context_node] = list(set(nodes.graph_nodes.context_nodes[context_node]))
            

        print('Getting their features...', '\n')
        index, node_feature_matrix, embeddings = nodes.get_node_features()
        print(f'Adding the edges of the word graph for the word "{target_word}"...')
        edges = Edges(
            index=index,
            nodes=nodes.graph_nodes,
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

        self.nodes.append(nodes.graph_nodes)
       
    
    def align_graphs(self) -> None:
        all_words = [set(self[i].index.key_to_index.keys()) for i in range(len(self.index))]
        dynamic_graph = is_dynamic(all_words)

        if not dynamic_graph:
            print('The graph nodes are static...', '\n')
            reference_index = self[0].index

            for i in range(1, len(self.index)):
                next_index = self[i].index
                index_mapping = {next_index.key_to_index[key]: reference_index.key_to_index[key] for key in next_index.key_to_index.keys()}

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
            reordered_index = GraphIndex(index_to_key=unified_dict_reverse, key_to_index=unified_dict)
             # {'index_to_key': unified_dict_reverse, 'key_to_index': unified_dict}

            for i in range(len(self.index)):
                snap_index = self[i].index
                index_mapping = {snap_index.key_to_index[key]: unified_dict[key] for key in snap_index.key_to_index.keys()}

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



if __name__ == '__main__':
    pass
