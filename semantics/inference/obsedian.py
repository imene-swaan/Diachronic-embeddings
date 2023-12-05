import os
from typing import List, Dict, Optional
from semantics.utils.utils import generate_pastel_colors
from semantics.utils.components import WordGraph
import matplotlib.pyplot as plt
import numpy as np





class ObsedianGraph:
    """
    Class for generating Obsedian graphs from WordGraphs.
    
    Args:
        vault_path (str): Path to the Obsedian vault.
        graph (WordGraph): The WordGraph from which to generate the Obsedian graph.

    Attributes:
        vault_path (str): Path to the Obsedian vault.
        index (Dict): Index of the graph.
        node_features (np.ndarray): Node features of the graph.
        edge_features (np.ndarray): Edge features of the graph.
        edge_index (np.ndarray): Edge indices of the graph.
        nodes (Dict[str, Dict]): Dictionary of nodes with their attributes. Created by _get_nodes().
        edges (Dict[str, Dict]): Dictionary of edges with their attributes. Created by _get_edges().
    """
    def __init__(
            self,
            vault_path: str,
            graph: WordGraph
            ):
        
        self.vault_path = vault_path
        if not os.path.exists(self.vault_path):
            raise ValueError(f"Vault path {self.vault_path} does not exist.")
        

        self.index = graph.index
        self.node_features = graph.node_features #node_types, node_levels, frequencies
        self.edge_features = graph.edge_features #edge_types, similarities, pmis, edge_strengths
        self.edge_index = graph.edge_index
    
        self.nodes = self._get_nodes()
        self.edges = self._get_edges()

    
    def generate_markdowns(self, folder: Optional[str] = '') -> None:
        """
        Generate markdown files for the nodes in the graph.
        
        Args:
            folder (str, optional): Folder in which to save the markdown files inside the vault.
        
        Examples:
            >>> obsedian_graph = ObsedianGraph()
            >>> obsedian_graph.generate_markdowns(folder= '2021-05-01')
        """
        if not os.path.exists(os.path.join(self.vault_path, folder)):
            os.mkdir(os.path.join(self.vault_path, folder))
        for node, node_attributes in self.nodes.items():
            node_path = os.path.join(self.vault_path, folder, f"{node}.md")
            node_edges = [val['content'] for key, val in self.edges.items() if node in key.split('-')]

            with open(node_path, "w") as f:
                f.write(self._generate_yaml_front_matter(node_attributes))
                f.write(f"# {node}\n\n")
                f.write("## Links\n\n")
                for edge in node_edges:
                    f.write(edge)

    def style(self) -> None:
        """Style the graph using the Obsedian CSS.
        
        Examples:
            >>> obsedian_graph = ObsedianGraph()
            >>> obsedian_graph.style()
        """
        graph_css_path = os.path.join(self.vault_path, ".obsidian", "plugins", "juggl", "graph.css")

        with open(graph_css_path, "w") as f:
            f.write('/* This is a custom CSS file for styling the graph view using juggl plugin. */\n\n')

            f.write("/* Node styles */\n")
            for node, node_attributes in self.nodes.items():
                f.write(f"node[title=\"{node}\"]" + "{\n")
                for attr, value in node_attributes.items():
                    f.write(f"\t{attr}: {value};\n")
                f.write("}\n\n")
            
            f.write("/* Edge styles */\n")
            for edge, edge_attributes in self.edges.items():
                f.write(f".type-{edge}" + "{\n")
                for attr, value in edge_attributes['style'].items():
                    f.write(f"\t{attr}: {value};\n")
                f.write("}\n\n")

    
    def _get_nodes(self) -> Dict[str, Dict]:
        """
        Get all the nodes in the graph.
        
        Returns:
            Dict[str, Dict]: Dictionary of nodes with their attributes.

        Examples:
            >>> obsedian_graph = ObsedianGraph()
            >>> obsedian_graph._get_nodes()
            {'a': {'title': 'a', 'color': (255, 127, 0), 'width': 250, 'height': 250}, 'b': {'title': 'b', 'color': (255, 127, 0), 'width': 250, 'height': 250}}
        """
        colors = generate_pastel_colors(np.unique(self.node_features[:, 0]).shape[0])
        color_map = {node_type: color for node_type, color in zip(np.unique(self.node_features[:, 0]), colors)}
        nodes = {}
        for key, idx in self.index['key_to_index'].items():
            nodes[key] = {
                'title': key,
                'color': color_map[self.node_features[idx, 0]],
                'width': self._float_to_size(self.node_features[idx, 2]),
                'height': self._float_to_size(self.node_features[idx, 2]),
            }
        return nodes

    def _get_edges(self) -> Dict[str, Dict]:
        """
        Get all the edges in the graph.
        
        Returns:
            Dict[str, Dict]: Dictionary of edges with their attributes.

        Examples:
            >>> obsedian_graph = ObsedianGraph()
            >>> obsedian_graph._get_edges()
            {'a-b': {'content': '- a-b [[b]]\\n', 'style': {'label': 0.5, 'line-color': (255, 127, 0), 'width': 250}}, 'b-a': {'content': '- b-a [[a]]\\n', 'style': {'label': 0.5, 'line-color': (255, 127, 0), 'width': 250}}}
        """
        edges = {}
        for i in range(self.edge_index.shape[1]):
            edge = self.edge_index[:, i]
            source = self.index['index_to_key'][edge[0]]
            target = self.index['index_to_key'][edge[1]]
            similarity = self.edge_features[i, 1]
            strength = self.edge_features[i, 3]

            edge_color = self._float_to_color(similarity)
            edge_width = self._float_to_size(strength)

            edge_content = f"- {source}-{target} [[{target}]]\n"
            edge_style = {
                'label': similarity,
                'line-color': edge_color,
                'width': edge_width,
            }
            edges[f"{source}-{target}"] = {
                'content': edge_content,
                'style': edge_style,
            }
        return edges
    
    def _generate_yaml_front_matter(self, attributes: Dict) -> str:
        """
        Generate the YAML front matter for the Obsedian note.
        
        Args:
            attributes (Dict): Dictionary of attributes for the note.
            
        Returns:
            str: YAML front matter.

        Examples:
            >>> obsedian_graph = ObsedianGraph()
            >>> obsedian_graph._generate_yaml_front_matter({'tags': 'test'})
            '---\\ntags: test\\n---\\n\\n'
        """
        yaml_content = "---\n"
        for attr, value in attributes.items():
            yaml_content += f"{attr}: {value}\n"
        yaml_content += "---\n\n"
        return yaml_content
    


    def _float_to_color(self, alpha: float, colormap=plt.cm.inferno) -> str:
        """
        Convert a value to a color using a specified colormap.
        
        Args:
            alpha (float): Value to convert to color.
            colormap (plt.cm, optional): Colormap to use. Defaults to plt.cm.inferno.
            
        Returns:
            str: Hex color code.
        
        Examples:
            >>> obsedian_graph = ObsedianGraph()
            >>> obsedian_graph._float_to_color(0.5)
            '#ff7f00'
        """
        return tuple(int(x * 255) for x in colormap(alpha)[:3])

    def _float_to_size(self, alpha: float) -> int:
        """
        Convert a value to a size in px. 
        
        Args:
            alpha (float): Value to convert to size.
            
        Returns:
            int: Size in px.
        
        Examples:
            >>> obsedian_graph = ObsedianGraph()
            >>> obsedian_graph._float_to_size(0.5)
            250
        """
        return int(alpha * 1000)/2        


if __name__ == "__main__":
    og = ObsedianGraph()

