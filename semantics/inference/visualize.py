import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from semantics.graphs.temporal_graph import TemporalGraph
from semantics.utils.components import WordGraph
from semantics.utils.utils import generate_colors
import numpy as np
from typing import Optional, Union, Tuple, Literal, List, Dict
import json
from collections import deque



def visualize_graph(
        graph: WordGraph,
        title: str = 'Graph Visualization',
        node_size_feature: Optional[int] = 0,
        node_color_feature: Optional[int] = 0,
        node_color_map: Optional[dict] = None,
        edge_label_feature: Optional[int] = 1,
        ax: Optional[plt.Axes] = None,
        color_norm: Optional[Union[Tuple[float, float], Literal['default', 'auto']]] = 'default',
        color_bar: bool = False,
        node_positions: Optional[Dict[int, np.ndarray]] = None,
        target_node: str = 'trump',
        radius: float = 2, # Distance from target node to level 1 nodes
        distance: float = 2 # Distance from level 1 nodes to level 2 nodes
    ) -> plt.Figure:
    """
    Visualize a graph.

    Args:
        graph (WordGraph): The graph to be visualized.
        title (str, optional): The title of the graph. Defaults to 'Graph Visualization'.
        node_size_feature (Optional[int], optional): The feature of the node to be used to scale the size of the nodes. Defaults to 0 (node_type: similar, context, target).
        node_color_feature (Optional[int], optional): The feature of the node to be used to color the nodes. Defaults to 0 (node_type: similar, context, target).
        node_color_map (Optional[dict], optional): The color map of the nodes. Defaults to None.
        edge_label_feature (Optional[int], optional): The feature of the edge to be used as label. Defaults to 1 (edge_type: similarity).
        ax (Optional[plt.Axes], optional): The axis of the figure. Defaults to None.
        color_norm (Optional[Union[Tuple[float, float], Literal['default', 'auto']]], optional): The normalization of the color bar. Defaults to 'default'.
        color_bar (bool, optional): Whether to show the color bar. Defaults to False.
        node_positions (Optional[Dict[int, np.ndarray]], optional): The positions of the nodes. Defaults to None.

    Returns:
        fig (plt.Figure): The figure of the graph.

    Examples:
        >>> from semantics.utils.components import WordGraph
        >>> from semantics.inference.visualize import visualize_graph
        >>> graph = WordGraph()
        >>> # add graph nodes and edges. See semantics/utils/components.py for more details.
        >>> fig = visualize_graph(graph, title='Graph Visualization', node_size_feature=0, node_color_feature=0, edge_label_feature=1)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 20))
    else:
        fig = ax.figure

    ax.set_title(title)
    
    nodes = [i for i in range(graph.node_features.shape[0]) if graph.node_features[i].sum() > 0]
    
    try: 
        node_labels = {i: graph.index.index_to_key[i] for i in nodes}
    except:
        node_labels = {i: graph.index.index_to_key[str(i)] for i in nodes}

    node_sizes = [5500 if graph.node_features[node, node_size_feature] == 1 else 
                    4000 if graph.node_features[node, node_size_feature] == 2 else 
                    2000 if graph.node_features[node, node_size_feature] == 3 else 
                    50 for node in nodes]
    
    if node_color_map is None:
        node_types = np.unique(graph.node_features[:, node_color_feature].tolist())
        colors = generate_colors(len(node_types), range_min = 0.7, range_max = 0.9)
        color_map = {int(val): color for val, color in zip(node_types, colors)}
        node_colors = [color_map[graph.node_features[node, node_color_feature]] for node in nodes]
    
    else:
        node_colors = [node_color_map[graph.node_features[node, node_color_feature]] for node in nodes]

    edges = list(map(tuple, graph.edge_index.T.tolist()))
    edge_labels = {(int(k[0]), int(k[1])): np.round(v, 2) for k, v in zip(edges, graph.edge_features[:, edge_label_feature])}

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for edge in edge_labels:
        G.add_edge(*edge, weight=edge_labels[edge])

    if node_positions is None:
        # pos = nx.spring_layout(G,  k=2)  # Node positions
        try:
            target_idx = graph.index.key_to_index[target_node]
        except:
            raise ValueError('Target node not found.')
        
        pos = create_position_template(graph=G, target_node_idx=target_idx)
    else:
        pos = {k:v for k, v in node_positions.items() if k in nodes}

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    cmap = plt.cm.plasma

    if color_norm == 'default':
        norm = mcolors.Normalize(vmin=0, vmax=1)
    elif color_norm == 'auto':
        norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    else:
        norm = mcolors.Normalize(vmin=color_norm[0], vmax=color_norm[1])
    
    edge_colors = [cmap(norm(weight)) for weight in edge_weights]
    edge_widths = [weight * 10 for weight in edge_weights]
    nx.draw(G, pos, with_labels=False, node_color= node_colors, edge_color=edge_colors,
                width=edge_widths, node_size=node_sizes, font_size=25)

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=30)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20)

    if color_bar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation= 'vertical', shrink=0.8, label= 'Similarity')
   
    ax.axis('off')
    return fig


def create_position_template(
        graph: nx.Graph, 
        target_node_idx: int,
        radius_level_1: float = 2,  # Distance from target node to level 1 nodes
        radius_level_2: float = 2,  # Distance from level 1 nodes to level 2 nodes
        radius_level_3: float = 1.5 # Distance from level 2 nodes to level 3 nodes
        ) -> Dict[int, np.ndarray]:
    # Start with an empty dictionary for positions
    pos = {}

    # Place the target node in the center
    pos[target_node_idx] = np.array([0.5, 0.5])

    # Get level 1 nodes (nodes directly connected to the target node)
    level_1_nodes = list(graph.neighbors(target_node_idx))

    # Calculate the positions of level 1 nodes in a circle around the target node
    angle_step = 2 * np.pi / len(level_1_nodes)
    for index, node in enumerate(level_1_nodes):
        angle = index * angle_step
        pos[node] = np.array([0.5 + radius_level_1 * np.cos(angle), 0.5 + radius_level_1 * np.sin(angle)])

    # Calculate positions for level 2 and level 3 nodes
    for l1_node in level_1_nodes:
        level_2_nodes = [n for n in graph.neighbors(l1_node) if n not in level_1_nodes and n != target_node_idx]
        
        angle_step_l2 = 2 * np.pi / max(1, len(level_2_nodes))  # Avoid division by zero
        l1_angle = np.arctan2(pos[l1_node][1] - 0.5, pos[l1_node][0] - 0.5)

        for index, l2_node in enumerate(level_2_nodes):
            angle = l1_angle + (index - len(level_2_nodes) / 2) * angle_step_l2 / 4  # Offset angle for level 2 nodes
            pos[l2_node] = pos[l1_node] + np.array([radius_level_2 * np.cos(angle), radius_level_2 * np.sin(angle)])

            # Handling level 3 nodes
            level_3_nodes = [n for n in graph.neighbors(l2_node) if n not in level_1_nodes and n not in level_2_nodes and n != target_node_idx]
            angle_step_l3 = 2 * np.pi / max(1, len(level_3_nodes))  # Avoid division by zero
            l2_angle = np.arctan2(pos[l2_node][1] - pos[l1_node][1], pos[l2_node][0] - pos[l1_node][0])

            for index_l3, l3_node in enumerate(level_3_nodes):
                angle_l3 = l2_angle + (index_l3 - len(level_3_nodes) / 2) * angle_step_l3 / 4
                pos[l3_node] = pos[l2_node] + np.array([radius_level_3 * np.cos(angle_l3), radius_level_3 * np.sin(angle_l3)])

    return pos


def create_position_template_max(graph: nx.Graph, target_node_idx: int, 
                             distance: float = 2, max_level: int = None) -> Dict[int, np.ndarray]:
    # Initialize position dict and queue for breadth-first traversal
    pos = {target_node_idx: np.array([0.5, 0.5])}
    queue = deque([(target_node_idx, 0)])  # Tuple of (node, level)
    # Set to keep track of placed nodes
    placed_nodes = {target_node_idx}

    while queue:
        current_node, current_level = queue.popleft()

        # Check if we have reached the maximum level
        if max_level is not None and current_level >= max_level:
            continue

        # Get neighbors not already placed
        neighbors = [n for n in graph.neighbors(current_node) if n not in placed_nodes]
        num_neighbors = len(neighbors)
        if num_neighbors == 0:
            continue

        angle_step = 2 * np.pi / num_neighbors
        base_angle = np.arctan2(pos[current_node][1] - 0.5, pos[current_node][0] - 0.5)

        for index, node in enumerate(neighbors):
            angle = base_angle + (index - num_neighbors / 2) * angle_step / 4
            pos[node] = pos[current_node] + np.array([0.5 + distance * np.cos(angle), 0.5 + distance * np.sin(angle)])
            placed_nodes.add(node)
            queue.append((node, current_level + 1))

    return pos

# def create_position_template(
#         graph: nx.Graph, 
#         target_node_idx: int,
#         radius: float = 2, # Distance from target node to level 1 nodes
#         distance: float = 2, # Distance from level 1 nodes to level 2 nodes
#         max_level: int = 2
#         ) -> Dict[int, np.ndarray]:
#     # Start with an empty dictionary for positions
#     pos = {}
    
#     print('\nTarget Node:', target_node_idx, '\n')
#     # Place the target node in the center
#     pos[target_node_idx] = np.array([0.5, 0.5])

#     # Get level 1 nodes (nodes directly connected to the target node)
#     level_1_nodes = list(graph.neighbors(target_node_idx))
#     num_level_1 = len(level_1_nodes)


#     # Calculate the positions of level 1 nodes in a circle around the target node
#     angle_step = 2 * np.pi / num_level_1
#     for index, node in enumerate(level_1_nodes):
#         print(node, '\n')
#         angle = index * angle_step
#         pos[node] = np.array([0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)])

#     # For each level 1 node, calculate the positions of level 2 nodes
#     for l1_node in level_1_nodes:
#         level_2_nodes = [n for n in graph.neighbors(l1_node) if n not in level_1_nodes and n != target_node_idx]
#         num_level_2 = len(level_2_nodes)
#         if num_level_2 == 0:
#             continue
#         angle_step = 2 * np.pi / num_level_2
#         l1_angle = np.arctan2(pos[l1_node][1] - 0.5, pos[l1_node][0] - 0.5)
        
#         for index, node in enumerate(level_2_nodes):
#             # print(node, '\n')
#             angle = l1_angle + (index - num_level_2 / 2) * angle_step / 4  # Offset angle for level 2 nodes
#             pos[node] = pos[l1_node] + np.array([distance * np.cos(angle), distance * np.sin(angle)])
#     return pos


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
            temporal_graph: TemporalGraph,
            title: str = 'Word Traffic',
            node_size_feature: int = 0,
            node_color_feature: int = 0,
            edge_label_feature: int = 1,
            node_color_map: Optional[dict] = None,
            target_node: str = 'trump',
            radius: float = 2, # Distance from target node to level 1 nodes
            distance: float = 2 # Distance from level 1 nodes to level 2 nodes
        ):
        """
        Initialize the WordTraffic object.
        
        Args:
            graph (TemporalGraph): The temporal graph to be visualized.
            title (str, optional): The title of the graph. Defaults to 'Word Traffic'.
            node_size_feature (int, optional): The feature of the node to be used to scale the size of the nodes. Defaults to 0 (node_type: similar, context, target).
            node_color_feature (int, optional): The feature of the node to be used to color the nodes. Defaults to 0 (node_type: similar, context, target).
            edge_label_feature (int, optional): The feature of the edge to be used as label. Defaults to 1 (edge_type: similarity).

        Examples:
            >>> from semantics.inference.visualize import WordTraffic
            >>> from semantics.graphs.temporal_graph import TemporalGraph
            >>> graph = TemporalGraph()
            >>> # add graph nodes and edges. See semantics/graphs/temporal_graph.py for more details.
            >>> word_traffic = WordTraffic(graph, title='Word Traffic', node_label_feature=0, edge_label_feature=1)
        """
        if edge_label_feature > temporal_graph[0].edge_features.shape[1]:
            raise ValueError('edge_label_feature out of range.')

        if node_size_feature > temporal_graph[0].node_features.shape[1]:
            raise ValueError('node_size_feature out of range.')
        
        if node_color_feature > temporal_graph[0].node_features.shape[1]:
            raise ValueError('node_color_feature out of range.')

        self.temporal_graph = temporal_graph
        self.title = title
        self.node_size_feature = node_size_feature
        self.node_color_feature = node_color_feature
        self.edge_label_feature = edge_label_feature
        self.fig, self.ax = plt.subplots(figsize=(30, 20))
        self.ax.set_title(self.title)

        self.sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=1))
        self.sm.set_array([])
        
        self.colorbar_created = False

        if edge_label_feature == 1:
            self.colorbar_label = 'Similarity'
        
        elif edge_label_feature == 2:
            self.colorbar_label = 'PMI'
        
        else:
            self.colorbar_label = 'Weight'
        
        if node_color_map is None:
            node_types = np.unique(temporal_graph[0].node_features[:, node_color_feature].tolist())
            colors = generate_colors(len(node_types), range_min = 0.7, range_max = 0.9)
            self.node_color_map = {int(val): color for val, color in zip(node_types, colors)}
        
        else:
            self.node_color_map = node_color_map
        
        index_to_key = list(temporal_graph[0].index.index_to_key.keys())

        G = nx.Graph()
        G.add_nodes_from(index_to_key)
        edge_index = []

        for graph in temporal_graph:
            edges = graph.edge_index.T.tolist()
            edges = list(map(tuple, edges))
            edges = list(map(lambda x: (int(x[0]), int(x[1])), edges))
            edge_index.extend(edges)

        edge_index = list(set(edge_index))
        for edge in edge_index:
            G.add_edge(*edge)

        target_idx = temporal_graph[0].index.key_to_index[target_node]
        self.all_nodes_positions = create_position_template(graph=G, target_node_idx=target_idx, radius=radius, distance=distance)

        del G
        del index_to_key
    
    def view(self, num):
        """
        View the graph at a specific time step.
        
        Args:
            num: The time step to be viewed.
        """
        # print(f'Viewing graph at time step {num}')
        self.ax.clear()
        current_graph = self.temporal_graph[num]

        
        visualize_graph(
            graph= current_graph, 
            title=self.title, 
            node_size_feature= self.node_size_feature,
            node_color_feature= self.node_color_feature,
            node_color_map= self.node_color_map,
            edge_label_feature=self.edge_label_feature, 
            ax=self.ax,
            color_norm='default',
            color_bar=False,
            node_positions=self.all_nodes_positions
            )
        
        
        
        if not self.colorbar_created:
            plt.colorbar(self.sm, ax=self.ax, orientation= 'vertical', shrink=0.8, label= self.colorbar_label)
            self.colorbar_created = True
    
    def animate(self, start: int = 0, end: int = None, repeat: bool = False, interval: int = 1000, save_path: Optional[str] = None):
        """
        Animate the graph from start to end time step.
        If save_path is not None, save the animation to the path.

        Args:
            start (int, optional): The start time step. Defaults to 0.
            end (int, optional): The end time step. Defaults to None.
            repeat (bool, optional): Whether to repeat the animation. Defaults to False.
            interval (int, optional): The interval between each frame. Defaults to 1000.
            save_path (Optional[str], optional): The path to save the animation. Defaults to None.
        
        Returns:
            anim (FuncAnimation): The animation.

        Examples:
            >>> from semantics.inference.visualize import WordTraffic
            >>> from semantics.graphs.temporal_graph import TemporalGraph
            >>> graph = TemporalGraph()
            >>> # add graph nodes and edges. See semantics/graphs/temporal_graph.py for more details.
            >>> word_traffic = WordTraffic(graph, title='Word Traffic', node_label_feature=0, edge_label_feature=1)
            >>> word_traffic.animate(start=0, end=10, repeat=False, interval=1000, save_path='word_traffic.gif')
        """
        if any([end is None, end == -1, end > len(self.temporal_graph.index)]):
            end = len(self.temporal_graph.index)

        
        anim = FuncAnimation(self.fig, self.view, frames=range(start, end), interval=interval, repeat=repeat)

        if save_path is not None:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='imagemagick')
            else:
                anim.save(save_path, writer='ffmpeg')
        return anim


    




if __name__ == '__main__':
    pass
