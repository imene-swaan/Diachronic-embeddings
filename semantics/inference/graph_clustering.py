from semantics.utils.components import WordGraph
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt



class GraphClusterer:
    """A class for clustering a word graph into semantically coherent groups."""

    def __init__(self, graph: WordGraph):

        self.index = graph.index
        self.node_features = graph.node_features # 
        self.edge_features = graph.edge_features
        self.edge_index = graph.edge_index
        self.labels = graph.labels
        self.label_mask = graph.label_mask

        nodes = list(map(int, list(self.index['index_to_key'].keys())))
        print('Nodes: ', nodes, '\n')
        print('Number of nodes: ', len(nodes), '\n')

        node_labels = dict(map(lambda item: (int(item[0]), item[1]), self.index['index_to_key'].items()))


        print('Node labels: ', node_labels, '\n')
        print('Node types: ', self.node_features[:, 0].tolist(), '\n')
        node_sizes = list(map(lambda x: int((int(x)+20)/(int(x)+1)), self.node_features[:, 0].tolist()))

        edges = self.edge_index.t().tolist()
        print(edges)
        raise NotImplementedError

        G = nx.Graph()

    def cluster(self):
        """Clusters the graph into semantically coherent groups."""
        raise NotImplementedError
