from semantics.utils.components import WordGraph
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from networkx.algorithms import community
import community as community_louvain
import numpy as np
from typing import Literal, Optional, Union, List
import itertools

def wordgraph_to_networkx(
        graph: WordGraph,
        edge_label_feature: Optional[int] = 1
    ) -> nx.Graph:
    

    
    node_indecies = [i for i in range(graph.node_features.shape[0]) if np.any(graph.node_features[i, :] > 0)]
    node_labels = {i: graph.index.index_to_key[i] for i in node_indecies}

    nodes = [(i, {'label': node_labels[i]}) for i in node_indecies]

    edges = list(map(tuple, graph.edge_index.T.tolist()))
    edge_labels = {(int(k[0]), int(k[1])): np.round(v, 2) for k, v in zip(edges, graph.edge_features[:, edge_label_feature])}

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for edge in edge_labels:
        G.add_edge(*edge, weight=edge_labels[edge])
    return G


class GraphClustering:
    def __init__(self, graph: Union[WordGraph, nx.Graph]):
        
        if isinstance(graph, WordGraph):
            self.graph = wordgraph_to_networkx(graph)

        elif isinstance(graph, nx.Graph):
            self.graph = graph
        else:
            raise ValueError(f'Unknown graph type: {type(graph)}')
        
        self.labels = {i: self.graph.nodes[i]['label'] for i in self.graph.nodes}
        print(self.graph)

    def get_clusters(
            self,
            method: Literal['louvain', 'kclique', 'girvan_newman', 'connected_components'] = 'girvan_newman',
            k: Optional[int] = None,
            label: bool = True,
            structure: bool = False
            ):
        
        if method == 'louvain':
            return self._louvain()
        elif method == 'kclique':
            return self._kclique(k, label, structure)
        elif method == 'girvan_newman':
            return self._girvan_newman(k=k, label=label, structure=structure)
        elif method == 'connected_components':
            return self._byConnectedComponents(label, structure)
        else:
            raise ValueError(f'Unknown clustering method: {method}')
    

    def DrawDendrogram(self):
        dendrogram = self._girvan_newman(k=None)
        pass

    
    def _StructureClustering(self, clustering: List[List[str]]):
        structure = {}

        for i in range(len(clustering)):
            structure[i+1] = []
            for word in clustering[i]:
                structure[i+1].append(word)
                
        return structure
 
    
        
    def _louvain(self):
        partition = community_louvain.best_partition(self.graph)
        # draw the graph
        pos = nx.spring_layout(self.graph)

        # color the nodes according to their partition
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                            cmap=cmap, node_color=list(partition.values()))
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()

        raise NotImplementedError
    
    def _kclique(self, k: int = 3, label: bool = True, structure: bool = False):
        kclique = community.k_clique_communities(self.graph, k=k)
        clusters = [list(c) for c in kclique]

        if label:
            clusters = [[self.labels[i] for i in c] for c in clusters]
        
        if structure:
            clusters = self._StructureClustering(clusters)
        
        return clusters
    
    def _girvan_newman(self, k: Optional[int] = None, label: bool = True, structure: bool = False):
        comp = community.girvan_newman(self.graph)

        dendrogram = []
        for communities in itertools.islice(comp, None):
            clusters = list(sorted(c) for c in communities)
            if label:
                labeled_clusters = [[self.labels[i] for i in c] for c in clusters]
                dendrogram.append(labeled_clusters)
            else:
                dendrogram.append(clusters)

        if k is not None:
            # find the clustering with k clusters or the last clustering
            lengths = [len(c) for c in dendrogram]
            diff = [abs(k - l) for l in lengths]
            min_diff_index = diff.index(min(diff))
            clustering = dendrogram[min_diff_index]
            
            if structure:
                clustering = self._StructureClustering(clustering)
                
            return clustering
        
        else:
            return dendrogram
            

    def _byConnectedComponents(self, label: bool = True, structure: bool = False):
        comp = list(nx.connected_components(self.graph))
        clusters = list(sorted(c) for c in comp)
        if label:
            clusters = [[self.labels[i] for i in c] for c in clusters]
        
        if structure:
            clusters = self._StructureClustering(clusters)
            
        return clusters
        
        
        





if __name__ == '__main__':
    G = nx.path_graph(10)
    graphclusterer = GraphClustering(G)
    graphclusterer.get_clusters(method='girvan_newman')
