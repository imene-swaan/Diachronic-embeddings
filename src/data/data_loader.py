
from typing import Union, List
from pathlib import Path
import xml.etree.ElementTree as ET 
from src.utils.utils import read_txt

class Loader():
    def __init__(
            self,
            texts: List[str]
            ):
        
        self.texts = texts
    

    @classmethod
    def from_txt(cls,
                 path: Union[str, Path]
                 ):
        return cls(read_txt(path))
    
    @classmethod
    def from_xml(cls,
                 path: Union[str, Path],
                 tag: str
                 ):
        
        tree = ET.parse(path)
        root = tree.getroot()
        texts = []
        for elem in root.findall('.//' + tag):
            if isinstance(elem.text, str):
                texts.append(elem.text)
        return cls(texts)
    
    def forward(self):
        return self.texts
        

