from semantics.utils.components import WordGraph
import networkx as nx
from networkx.algorithms import community
import community as community_louvain
import numpy as np
from typing import Literal, Optional, Union
import itertools

def wordgraph_to_networkx(
        graph: WordGraph,
        edge_label_feature: Optional[int] = 1
    ) -> nx.Graph:
    

    
    nodes = [i for i in range(graph.node_features.shape[0]) if graph.node_features[i].sum() > 0]
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
        

    def get_clusters(
            self,
            method: Literal['louvain', 'kclique', 'girvan_newman'] = 'louvain',
            ):
        
        if method == 'louvain':
            return self._louvain()
        elif method == 'kclique':
            return self._kclique()
        elif method == 'girvan_newman':
            return self._girvan_newman()
        else:
            raise ValueError(f'Unknown clustering method: {method}')
        
    def _louvain(self):
        partition = community_louvain.best_partition(self.graph)
        raise NotImplementedError
    
    def _kclique(self):
        kclique = community.k_clique_communities(self.graph, 3)
        raise NotImplementedError
    
    def _girvan_newman(self):
        communities = community.girvan_newman(self.graph)
        limited = itertools.takewhile(lambda c: len(c) <= 4, communities)
        print(list(limited))
        # for communities in limited:
        #     print(tuple(sorted(c) for c in communities))




if __name__ == '__main__':
    
    G = nx.path_graph(10)
    graphclusterer = GraphClustering(G)
    graphclusterer.get_clusters(method='girvan_newman')
