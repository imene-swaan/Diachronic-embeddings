import os
import random
import networkx as nx
from pathlib import Path
from typing import List, Dict, Union, Optional
from semantics.graphs.temporal_graph import TemporalGraph
from semantics.utils.components import WordGraph








class ObsedianGraph:
    def __init__(
            self,
            vault_path: str,
            graph: Union[TemporalGraph, WordGraph],
            groups: Optional[List[str]] = None,
            ):
        
        self.vault_path = vault_path

        if isinstance(graph, TemporalGraph):
            wgs = []
            for g in range(len(graph.snapshots)):
                wgs.append(graph[g])
        
        elif isinstance(graph, WordGraph):
            self.wg = graph
        
        else:
            raise ValueError('graph must be either a TemporalGraph or a WordGraph')

        if groups is None:
            self.groups = ['low', 'medium', 'high']
        
        else:
            self.groups = groups

    
    def 
