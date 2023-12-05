import os
from typing import List, Dict, Union, Optional
from semantics.utils.components import WordGraph
from semantics.graphs.temporal_graph import TemporalGraph
import matplotlib.pyplot as plt






class ObsedianGraph:
    def __init__(
            self,
            # vault_path: str,
            # graph: WordGraph,
            groups: Optional[List[str]] = None,
            ):
        
        # self.vault_path = vault_path
        # self.graph = graph
        self.groups = groups if groups is not None else ['low', 'medium', 'high']


    
    def _generate_yaml_front_matter(self, attributes: Dict) -> str:
            yaml_content = "---\n"
            for attr, value in attributes.items():
                yaml_content += f"{attr}: {value}\n"
            yaml_content += "---\n\n"
            return yaml_content

   


    def _float_to_color(self, alpha: float, colormap=plt.cm.inferno) -> str:
        """Convert a value to a color using a specified colormap."""
        return tuple(int(x * 255) for x in colormap(alpha)[:3])

    def _float_to_size(self, alpha: float) -> int:
        """Convert a value to a size."""
        return int(alpha * 1000)


if __name__ == "__main__":
    og = ObsedianGraph()

