from cv2 import norm
from regex import E
from semantics.utils.components import WordGraph
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors












import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class GraphClusterer:
    """A class for clustering a word graph into semantically coherent groups."""

    def __init__(self, graph):
        self.graph = graph  # Assuming 'graph' is an instance of WordGraph
        self.construct_graph()

    def construct_graph(self):
        nodes = [i for i in range(self.graph.node_features.shape[0]) if self.graph.node_features[i].sum() > 0]
        node_labels = {i: self.graph.index['index_to_key'][str(i)] for i in nodes}

        node_sizes = [3500 if self.graph.node_features[node, 0] == 1 else 
                      2000 if self.graph.node_features[node, 0] == 2 else 
                      1000 if self.graph.node_features[node, 0] == 3 else 
                      50 for node in nodes]
        
        print('Node sizes: ', node_sizes)

        edges = list(map(tuple, self.graph.edge_index.T.tolist()))
        edge_labels = {(int(k[0]), int(k[1])): np.round(v, 2) for k, v in zip(edges, self.graph.edge_features[:, 1])}

        G = nx.Graph()
        G.add_nodes_from(nodes)
        for edge in edge_labels:
            G.add_edge(*edge, weight=edge_labels[edge])

        pos = nx.spring_layout(G)  # Node positions
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

        cmap = plt.cm.plasma
        # norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        norm = mcolors.Normalize(vmin=0, vmax=1)
        edge_colors = [cmap(norm(weight)) for weight in edge_weights]
        edge_widths = [weight * 5 for weight in edge_weights]

        self.draw_graph(G, pos, node_labels, node_sizes, edge_colors, edge_widths, edge_labels)

    def draw_graph(self, G, pos, node_labels, node_sizes, edge_colors, edge_widths, edge_labels):
        plt.figure(figsize=(12, 12))
        nx.draw(G, pos, with_labels=False, node_color='skyblue', edge_color=edge_colors,
                width=edge_widths, node_size=node_sizes, font_size=15)

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        ax = plt.gca()
        ax.set_axis_off()
        plt.savefig('graph.png')
        plt.show()



















# class GraphClusterer:
#     """A class for clustering a word graph into semantically coherent groups."""

#     def __init__(self, graph: WordGraph):

#         self.index = graph.index
#         self.node_features = graph.node_features # 
#         self.edge_features = graph.edge_features
#         self.edge_index = graph.edge_index
#         self.labels = graph.labels
#         self.label_mask = graph.label_mask

#         nodes = [i for i in range(self.node_features.shape[0]) if self.node_features[i].sum() > 0]
#         node_labels = {i: self.index['index_to_key'][str(i)] for i in nodes}

#         node_sizes = []
#         for node in nodes:
#             node_type = self.node_features[node, 0]
#             if node_type == 1:
#                 node_sizes.append(100)
#             elif node_type == 2:
#                 node_sizes.append(50)
#             elif node_type == 3:
#                 node_sizes.append(30)
            
#             else:
#                 node_sizes.append(10)

#         node_sizes = [ns for i, ns in enumerate(node_sizes) if i in nodes]

#         edges = list(map(tuple, self.edge_index.T.tolist()))
#         edge_labels = {(int(k[0]), int(k[1])):np.round(v,2) for k, v in zip(edges, self.edge_features[:, 1].tolist())}
        
#         cmap = plt.cm.plasma
#         edge_colors = [cmap(v) for _, v in edge_labels.items()]
#         edge_width = [v*5 for _, v in edge_labels.items()]
        
#         G = nx.Graph()
#         G.add_nodes_from(nodes)

#         for edge in edge_labels.keys():
#             G.add_edge(edge[0], edge[1], weight=edge_labels[edge])

#         weights = [G[u][v]['weight'] for u,v in G.edges()]
#         cmap = plt.cm.plasma
#         norm = mcolors.Normalize(vmin=0, vmax=1)
#         edge_colors = [cmap(norm(weight)) for weight in weights]

#         pos = nx.spring_layout(G)  # Node positions
        
#         nx.draw(
#             G, 
#             with_labels=False, 
#             node_color='skyblue', 
#             edge_color=edge_colors, 
#             width=edge_width, 
#             node_size = node_sizes, 
#             font_size= 15, 
#             pos= pos
#             )
        
#         nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

#         # pc = mpl.collections.PatchCollection(G.edges(), cmap=cmap)
#         # pc.set_array(edge_colors)

#         ax = plt.gca()
#         ax.set_axis_off()
#         # plt.colorbar(pc, ax=ax)
#         plt.savefig('graph.png')


#         # G.add_edges_from(edges)
#         # pos = nx.spring_layout(G)

#         # n = nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
#         # e = nx.draw_networkx_edges(
#         #     G, 
#         #     pos,
#         #     node_size=node_sizes,
#         #     arrowstyle='->',
#         #     arrowsize=10,
#         #     edge_color=edge_colors,
#         #     width=edge_width,
#         #     )
        
#         # nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

#         # for i in range(G.number_of_edges()):
#         #     e[i].set_alpha(edge_labels[e[i]])

#         # pc = mpl.collections.PatchCollection(e, cmap=cmap)
#         # pc.set_array(edge_colors)

#         # ax = plt.gca()
#         # ax.set_axis_off()
#         # plt.colorbar(pc, ax=ax)
#         # plt.savefig('graph.png')

#     def cluster(self):
#         """Clusters the graph into semantically coherent groups."""
#         raise NotImplementedError
