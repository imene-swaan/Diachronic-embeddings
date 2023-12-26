from click import Option
from semantics.feature_extraction.roberta import RobertaInference
from semantics.feature_extraction.bert import BertInference
from semantics.feature_extraction.word2vec import Word2VecInference
from typing import List, Union, Dict, Optional, Tuple
import torch
import numpy as np
from math import log
from semantics.utils.utils import count_occurence, most_frequent
import tqdm
from semantics.utils.components import WordGraph, GraphNodes, TargetWords, GraphIndex
from pydantic import ValidationError



class Nodes:
    """
    This class is used to structure the nodes of the word graph.

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
            target: Union[str, List[str], Dict[str, List[str]]],
            dataset: List[str],
            level: int,
            k: int,
            c: int,
            word2vec_model: Word2VecInference,
            mlm_model: Union[RobertaInference, BertInference],
            keep_k: Optional[Dict[int, Tuple[int, int]]] = None
            ):
        
        """
        """

        
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

        try:
            TargetWords(words=target)
        
        except ValidationError:
            raise ValueError('The target word must be a string, a list of strings, or a dictionary. Check the TargetWords class for more information.')
        
        else:
            self.target = target
        
    

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
    
    def get_centered_nodes(
            self,
            # target: str
            ) -> GraphNodes:
        """
        This method is used to get the nodes of the word graph (similar nodes, context nodes, and target node).

        Returns:
            graph_nodes (GraphNodes): the nodes of the word graph. Contains similar_nodes and context_nodes dictionaries. Check the GraphNodes class for more information.

        Examples:
            >>> word2vec = Word2VecInference('word2vec.model')
            >>> mlm = RobertaInference('MLM_roberta')
            >>> nd = Nodes(target_word='sentence', dataset=['this is a sentence', 'this is another sentence'], level=3, k=2, c=2, word2vec_model = word2vec, mlm_model = mlm)
            >>> graph_nodes = nd.get_nodes()
            >>> print(graph_nodes.similar_nodes)
            {'sentence': ['sentence', 'sentence'], 'this': ['this', 'this'], 'is': ['is', 'is'], 'a': ['a', 'a'], 'another': ['another', 'another']}
            >>> print(graph_nodes.context_nodes)
            {'sentence': ['this', 'is'], 'this': ['sentence', 'is'], 'is': ['this', 'sentence'], 'a': ['this', 'is'], 'another': ['this', 'is']}
        """
        if self.level == 0:
            raise ValueError('The level of the centered graph must be greater than 0.')
        mlm_nodes = {}
        word2vec_nodes = {}
        for level in range(self.level):
            print(f'Getting the nodes of level {level} ...')

            if level == 0:
                similar_nodes  = self.get_similar_nodes(self.target, keep_k= self.keep_k[level][0])
                context_nodes = self.get_context_nodes(self.target, keep_k= self.keep_k[level][1])

                mlm_nodes = similar_nodes
                word2vec_nodes = context_nodes

            else:

                previous_nodes = [node for node_list in mlm_nodes.values() for node in node_list if node not in mlm_nodes.keys()] + [node for node_list in word2vec_nodes.values() for node in node_list if node not in word2vec_nodes.keys()]
            
                previous_nodes = list(set(previous_nodes))

                similar_nodes = self.get_similar_nodes(previous_nodes, keep_k= self.keep_k[level][0])
                context_nodes = self.get_context_nodes(previous_nodes, keep_k= self.keep_k[level][1])

                mlm_nodes.update(similar_nodes)
                word2vec_nodes.update(context_nodes)

        graph_nodes = GraphNodes(
            similar_nodes= mlm_nodes,
            context_nodes= word2vec_nodes
        )
        return graph_nodes
        
    def get_circular_nodes(self) -> GraphNodes:
        target = list(set(self.target))
        mlm_nodes = {w: target[:i] + target[i:] for i, w in enumerate(target)}
        word2vec_nodes = {}

        if self.level == 0:
            graph_nodes = GraphNodes(
                similar_nodes= mlm_nodes
            )
            return graph_nodes

        else:
           
            for level in range(self.level):
                # print(f'Getting the nodes of level {level} ...')

                if level == 0:
                    similar_nodes  = self.get_similar_nodes(target, keep_k= self.keep_k[level][0])
                    context_nodes = self.get_context_nodes(target, keep_k= self.keep_k[level][1])

                    for w in mlm_nodes.keys():
                        if w in similar_nodes.keys():
                            mlm_nodes[w] += similar_nodes[w]
                            del similar_nodes[w]

                    mlm_nodes.update(similar_nodes)
                    word2vec_nodes.update(context_nodes)

                else:

                    previous_nodes = [node for node_list in mlm_nodes.values() for node in node_list if node not in mlm_nodes.keys()] + [node for node_list in word2vec_nodes.values() for node in node_list if node not in word2vec_nodes.keys()]
                
                    previous_nodes = list(set(previous_nodes))

                    similar_nodes = self.get_similar_nodes(previous_nodes, keep_k= self.keep_k[level][0])
                    context_nodes = self.get_context_nodes(previous_nodes, keep_k= self.keep_k[level][1])

                    mlm_nodes.update(similar_nodes)
                    word2vec_nodes.update(context_nodes)

            graph_nodes = GraphNodes(
                similar_nodes= mlm_nodes,
                context_nodes= word2vec_nodes
            )
            return graph_nodes
        
    def get_hierarchical_nodes(self) -> GraphNodes:
        target = list(self.target.keys())
        mlm_nodes = {w: target[:i] + target[i:] for i, w in enumerate(target)}
        word2vec_nodes = {}
        if self.level == 0:
            graph_nodes = GraphNodes(
                similar_nodes= mlm_nodes
            )
            return graph_nodes


        for level in range(self.level):
            if level == 0:
                previous_nodes = sum(list(self.target.values()), [])
                similar_nodes  = self.get_similar_nodes(previous_nodes, keep_k= self.keep_k[level][0])
                context_nodes = self.get_context_nodes(previous_nodes, keep_k= self.keep_k[level][1])

                for w in mlm_nodes.keys():
                    if w in similar_nodes.keys():
                        mlm_nodes[w] += similar_nodes[w]
                        del similar_nodes[w]

                mlm_nodes.update(similar_nodes)
                word2vec_nodes.update(context_nodes)

            else:
                    
                    previous_nodes = [node for node_list in mlm_nodes.values() for node in node_list if node not in mlm_nodes.keys()] + [node for node_list in word2vec_nodes.values() for node in node_list if node not in word2vec_nodes.keys()]
                
                    previous_nodes = list(set(previous_nodes))
    
                    similar_nodes = self.get_similar_nodes(previous_nodes, keep_k= self.keep_k[level][0])
                    context_nodes = self.get_context_nodes(previous_nodes, keep_k= self.keep_k[level][1])
    
                    mlm_nodes.update(similar_nodes)
                    word2vec_nodes.update(context_nodes)

        graph_nodes = GraphNodes(
            similar_nodes= mlm_nodes,
            context_nodes= word2vec_nodes
        )
        return graph_nodes



    def get_nodes(self) -> GraphNodes:
        if isinstance(self.target, str):
            self.nodes = self.get_centered_nodes()
            self.target = [self.target]
        
        elif isinstance(self.target, list):
            self.nodes = self.get_circular_nodes()

        else:
            self.nodes = self.get_hierarchical_nodes()
            self.target = list(self.target.keys())

        return self.nodes  
        
        
    def get_node_features(self) -> Tuple[GraphIndex, np.ndarray, np.ndarray]:
        """
        
        """
        
            
        words = self.target
        node_types = [1]*len(words)
        all_words_count = count_occurence(self.dataset)
        frequencies = [count_occurence(self.dataset, word)/ all_words_count for word in words]
        embeddings = [self.mlm.get_embedding(main_word=word).mean(axis=0) for word in words]

        for node_list in self.nodes.similar_nodes.values():
            for node in node_list:
                if node in words:
                    continue

                words.append(node)
                node_types.append(2)
                frequencies.append(count_occurence(self.dataset, node) / all_words_count)
                embeddings.append(self.mlm.get_embedding(main_word=node).mean(axis=0))

        for node_list in self.nodes.context_nodes.values():
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

        self.graph_nodes: List[GraphNodes] = []
        
    
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
            ) -> GraphNodes:
        """
        """

        # print(f'Adding the nodes of the word graph for the word "{target_word}"...')
        nd = Nodes(
            target= target_word,
            dataset=dataset,
            level= level,
            k= k,
            c= c,
            word2vec_model = word2vec_model,
            mlm_model = mlm_model,
            keep_k= keep_k
            )

        nodes = nd.get_nodes()

        if accumulate and (len(self.graph_nodes) > 0):
            print('Accumulating the nodes of the word graph...')
            previous_nodes = self.graph_nodes[-1]

            for similar_node in previous_nodes.similar_nodes.keys():
                if similar_node not in nodes.similar_nodes.keys():
                    nodes.similar_nodes[similar_node] = previous_nodes.similar_nodes[similar_node]
                else:
                    nodes.similar_nodes[similar_node] += previous_nodes.similar_nodes[similar_node]
                    nodes.similar_nodes[similar_node] = list(set(nodes.similar_nodes[similar_node]))

            for context_node in previous_nodes.context_nodes.keys():
                if context_node not in nodes.context_nodes.keys():
                    nodes.context_nodes[context_node] = previous_nodes.context_nodes[context_node]
                else:
                    nodes.context_nodes[context_node] += previous_nodes.context_nodes[context_node]
                    nodes.context_nodes[context_node] = list(set(nodes.context_nodes[context_node]))
            

        print('Getting their features...', '\n')
        index, node_feature_matrix, embeddings = nd.get_node_features()
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

        self.graph_nodes.append(nodes)
        return nodes
    
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


