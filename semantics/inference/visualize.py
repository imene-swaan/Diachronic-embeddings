import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from semantics.graphs.temporal_graph import TemporalGraph
from semantics.utils.components import WordGraph
from semantics.utils.utils import generate_pastel_colors
import numpy as np
from typing import Optional, Union



def visualize_graph(
        graph: Optional[Union[tuple, WordGraph]] = None,
        title: str = 'Graph Visualization',
        node_label_feature: Optional[int] = 0,
        edge_label_feature: Optional[int] = 1,
        ax: Optional[plt.Axes] = None
    ):
    """
    Visualize a graph.

    Args:
        graph (Optional[Union[tuple, WordGraph]]): The graph to be visualized. If None, return an empty figure. Default to None. If tuple, the graph is a temporal word graph. If WordGraph, the graph is a word graph.
        title: The title of the graph. Default to 'Graph Visualization'.
        node_label_feature: The feature of the node to be used as label. Default to 0 (node_type).
        edge_label_feature: The feature of the edge to be used as label. Default to 1 (Similarities).
        ax: The axis of the figure. If None, create a new figure.
    
    Returns:
        fig: The figure of the graph.

    Examples:
        >>> from semantics.inference.visualize import visualize_graph
        >>> from semantics.graphs.temporal_graph import TemporalGraph
        >>> graph = TemporalGraph()
        >>> # add graph nodes and edges. See semantics/graphs/temporal_graph.py for more details.
        >>> fig = visualize_graph(graph, title='Graph Visualization', node_label_feature=0, edge_label_feature=1)
        >>> fig.savefig('graph.png')
        >>> # or
        >>> fig.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 12))
    else:
        fig = ax.figure

    ax.set_title(title)
   
    if graph is None:
        return fig
    
    index, xs, edge_indices, edge_features, _, _ = graph

    node_feature = xs[:, node_label_feature]
    unique_values = np.unique(node_feature, return_counts=False)
    random_pastel_colors = generate_pastel_colors(len(unique_values))
    feature_to_color_map = dict(zip(unique_values, random_pastel_colors))

    node_labels = list(index['key_to_index'].keys())
    node_colors = [feature_to_color_map[node_feature[i]] for i in range(len(node_feature))]
    color_map = dict(zip(node_labels, node_colors))

    G = nx.Graph()
    G.add_nodes_from(node_labels)

    for i in range(edge_indices.shape[1]):
        G.add_edge(
            index['index_to_key'][edge_indices[0][i]],
            index['index_to_key'][edge_indices[1][i]],
            weight= edge_features[i][edge_label_feature]
            )

    weights = [G[u][v]['weight'] for u, v in G.edges()]
    cmap = plt.get_cmap('coolwarm')  # Blue to red colormap
    norm = mcolors.Normalize(vmin=0, vmax=1)
    edge_colors = [cmap(norm(weight)) for weight in weights]

    # pos = nx.spring_layout(G)  # Node positions
    nx.draw(G,  ax= ax, with_labels=True, node_color=[color_map[node] for node in G.nodes()], edge_color=edge_colors, width=2, node_size = 1000, font_size= 15) #pos= pos,

    # Edge labels (weights)
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    ax.axis('off')
    return fig



class WordTraffic:
    """
    Visualize the word traffic of a temporal word graph.
    
    Methods:
        __init__(graph, title, node_label_feature, edge_label_feature)
            Initialize the WordTraffic object.
        view(num)
            View the graph at a specific time step.
        animate(start, end, repeat, interval, save_path)
            Animate the graph from start to end time step.
            If save_path is not None, save the animation to the path.
    """
    def __init__(
            self,
            graph: TemporalGraph,
            title: str = 'Word Traffic',
            node_label_feature: Optional[int] = None,
            edge_label_feature: Optional[int] = 1
        ):
        """
        Initialize the WordTraffic object.
        
        Args:
            graph: The temporal word graph.
            title: The title of the graph.
            node_label_feature: The feature of the node to be used as label.
            edge_label_feature: The feature of the edge to be used as label.
        
        Attributes:
            graph: The temporal word graph.
            title: The title of the graph.
            node_label_feature: The feature of the node to be used as label.
            edge_label_feature: The feature of the edge to be used as label.
            fig: The figure of the graph.
            ax: The axis of the figure.
        """
        self.graph = graph
        self.title = title
        self.node_label_feature = node_label_feature
        self.edge_label_feature = edge_label_feature
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title(self.title)
    
    def view(self, num):
        """
        View the graph at a specific time step.
        
        Args:
            num: The time step to be viewed.
        """
        self.ax.clear()
        graph = self.graph[num]
        visualize_graph(graph, title=self.title, node_label_feature=self.node_label_feature, edge_label_feature=self.edge_label_feature, ax=self.ax)

    
    def animate(self, start: int = 0, end: int = None, repeat: bool = False, interval: int = 1000, save_path: Optional[str] = None):
        """
        Animate the graph from start to end time step.
        If save_path is not None, save the animation to the path.

        Args:
            start: The start time step.
            end: The end time step.
            repeat: Whether to repeat the animation.
            interval: The interval between each frame.
            save_path: The path to save the animation.

        Returns:
            anim: The animation.

        Examples:
            >>> from semantics.inference.visualize import WordTraffic
            >>> from semantics.graphs.temporal_graph import TemporalGraph
            >>> graph = TemporalGraph()
            >>> # add graph nodes and edges. See semantics/graphs/temporal_graph.py for more details.
            >>> word_traffic = WordTraffic(graph, title='Word Traffic', node_label_feature=0, edge_label_feature=1)
            >>> word_traffic.animate(start=0, end=10, repeat=True, interval=1000, save_path='word_traffic.gif')
        """
        if end is None:
            end = len(self.graph)
        
        anim = FuncAnimation(self.fig, self.view, frames=range(start, end), interval=interval, repeat=repeat)

        if save_path is not None:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='imagemagick')
            else:
                anim.save(save_path, writer='ffmpeg')
        return anim


    




if __name__ == '__main__':
    visualize_graph().savefig('test.png')
