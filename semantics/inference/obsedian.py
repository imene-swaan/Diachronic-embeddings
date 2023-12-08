import os
from typing import List, Dict, Optional, Tuple
from semantics.utils.utils import generate_pastel_colors
from semantics.utils.components import WordGraph
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
            vault_path: Optional[str] = None,
            graph: Optional[WordGraph] = None,
            ):
        self.vault_path = vault_path
        if not os.path.exists(self.vault_path):
            raise ValueError(f"Vault path {self.vault_path} does not exist.")
        

        self.index = graph.index
        self.node_features = graph.node_features #node_types, node_levels, frequencies
        self.edge_features = graph.edge_features #edge_types, similarities, pmis, edge_strengths
        self.edge_index = graph.edge_index
    
        self.nodes, self.tag_styles = self._get_nodes()
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
        

        for node, node_content in self.nodes.items():
            node_path = os.path.join(self.vault_path, folder, f"{node}.md")
            node_edges = [val['content'] for key, val in self.edges.items() if node == key.split('AND')[0]]

            with open(node_path, "w") as f:
                # f.write(self._generate_node_tags(node_attributes['content']))
                f.write(node_content)
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
            for tag, style in self.tag_styles.items():
                f.write(f".tag-{tag}" + "{\n")
                for attr, value in style.items():
                    f.write(f"\t{attr}: {value};\n")
                f.write("}\n\n")

            
            f.write("/* Edge styles */\n")
            for edge, edge_attributes in self.edges.items():
                f.write(f".type-{edge}" + "{\n")
                for attr, value in edge_attributes['style'].items():
                    f.write(f"\t{attr}: {value};\n")
                f.write("}\n\n")

    
    def _get_nodes(self) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """
        Get all the nodes in the graph.
        
        Returns:
            Dict[str, Dict]: Dictionary of nodes with their attributes.
        """
        nodes = {}
        unique_types = set()
        unique_levels = set()
        for key, idx in self.index['key_to_index'].items():
            type_tag = f"type{int(self.node_features[idx, 0])}" 
            level_tag = f"level{int(self.node_features[idx, 1])}" 
            unique_types.add(type_tag)
            unique_levels.add(level_tag) 
            nodes[key] = self._format_tags(tags= [type_tag, level_tag])           
        

        tag_style = {}
        
        colors = ["#b4f927", "#13ebef", "#1118f0"]   # generate_pastel_colors(len(unique_types))
        color_map = {val: color for val, color in zip(list(unique_types), colors)}
        for tag in unique_types:
            tag_style[tag] = {
                'background-color': color_map[tag]
            }


        sizes = self._category_to_size(unique_levels)
        size_map = {val: size for val, size in zip(list(unique_levels), sizes)}
        for tag in unique_levels:
            size = self._float_to_size(size_map[tag], scale_max=20)
            tag_style[tag] = {
                'width': f"{size}px",
                'height': f"{size}px"
            }

        return nodes, tag_style

    def _category_to_size(self, categories: list) -> List[str]:
        items = len(categories)
        return [np.round(i+1/items, 2) for i in range(items)]

    
    def _format_tags(self, tags: List[str]) -> str:
        content = "\n"
        for tag in tags:
            content += f"#{tag}\n"
        content += "\n\n"
        return content


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
            edge_width = self._float_to_size(similarity, scale_max=10)

            edge_content = f"- {source}AND{target} [[{target}]]\n"
            edge_style = {
                'label': str(np.round(similarity, 2)),
                'line-color': edge_color,
                'target-arrow-color': edge_color,
                'width': edge_width,
            }
            edges[f"{source}AND{target}"] = {
                'content': edge_content,
                'style': edge_style,
            }
        return edges
    
    # def _generate_yaml_front_matter(self, attributes: Dict) -> str:
    #     yaml_content = "---\n"
    #     for attr, value in attributes.items():
    #         yaml_content += f"{attr}: {value}\n"
    #     yaml_content += "---\n\n"
    #     return yaml_content
    


    def _float_to_color(self, alpha: float, colormap=plt.cm.get_cmap('inferno_r')) -> str:
        """
        Convert a value to a color using a specified colormap.
        
        Args:
            alpha (float): Value to convert to color. Should be between 0 and 1.
            colormap (plt.cm, optional): Colormap to use. Defaults to plt.cm.get_cmap('inferno_r').
            
        Returns:
            str: Hex color code.
        
        Examples:
            >>> obsedian_graph = ObsedianGraph()
            >>> obsedian_graph._float_to_color(0.5)
            '#ff7f00'
        """
        return mcolors.to_hex(colormap(alpha/1.5))

    def _float_to_size(self, alpha: float, scale_min: int = 0, scale_max:int = 50) -> int:
        """
        Convert a value to a size.
        
        Args:
            alpha (float): Value to convert to size. Should be between 0 and 1.
            scale_min (int): Minimum size. Defaults to 0.
            scale_max (int): Maximum size. Defaults to 50.
            
        Returns:
            int: Size.
        
        Examples:
            >>> obsedian_graph = ObsedianGraph()
            >>> obsedian_graph._float_to_size(0.5, 0, 50)
            25
        """
        return int(alpha * (scale_max - scale_min) + scale_min)      


if __name__ == "__main__":
    # og = ObsedianGraph()

    # print(og._float_to_size(0.5, 0, 10))

    pass
