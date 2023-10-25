
from typing import Union, List
from pathlib import Path
import xml.etree.ElementTree as ET 
from src.utils.utils import read_txt, sample_data
import os


class Loader():
    def __init__(
            self,
            texts: List[str]
            ):
        """
        Class for loading data.

        Args:
            texts (List[str]): List of texts.
        
        Methods:
            from_txt: Reads texts from a text file.
            from_xml: Reads texts from an XML file.
            forward: Filters the texts based on the target words and the maximum number of documents.
            split_xml: Splits an XML file into multiple XML files with a maximum number of children.
        """
        
        self.texts = texts
    

    @classmethod
    def from_txt(cls,
                 path: Union[str, Path]
                 ):
        """
        Reads texts from a text file.

        Args:
            path (Union[str, Path]): Path to the text file.
        """
        return cls(read_txt(path))
    
    @classmethod
    def from_xml(cls,
                 path: Union[str, Path],
                 tag: str
                 ):
        """
        Reads texts from an XML file.
        
        Args:
            path (Union[str, Path]): Path to the XML file.
            tag (str): Tag of the XML file.
        """
        size = os.path.getsize(path)
        if size > 1e8:
            raise ValueError("File size is too large. Please split the file into smaller files.")
        tree = ET.parse(path)
        root = tree.getroot()
        texts = []
        for elem in root.findall('.//' + tag):
            if isinstance(elem.text, str):
                texts.append(elem.text)
        return cls(texts)
    

    
    def forward(self, target_words: List[str] = None, max_documents: int = None, shuffle: bool = True, random_seed=None) -> List[str]:
        """
        Filters the texts based on the target words and the maximum number of documents.

        Args:
            target_words (List[str], optional): List of target words. Defaults to None.
            max_documents (int, optional): Maximum number of documents. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            random_seed ([type], optional): Random seed. Defaults to None.

        Returns:
            List[str]: List of texts.
        """

        if target_words is not None:
            relevant_texts = []
            for text in self.texts:
                if any([' ' + word + ' ' in text for word in target_words]):
                    relevant_texts.append(text)
            
            self.texts = relevant_texts
        
        if max_documents is not None:
            if shuffle:
                self.texts = sample_data(self.texts, max_documents, random_seed)
            else:
                self.texts = self.texts[:max_documents]

        return self.texts
    




     
        

def split_xml(path:str, output_dir:str, max_children:int = 1000) -> List[str]:
    """
    Splits an XML file into multiple XML files with a maximum number of children.

    Args:
        path (str): Path to the XML file.
        output_dir (str): Path to the output directory.
        max_children (int, optional): Maximum number of children. Defaults to 1000.

    Returns:
        List[str]: List of paths to the new XML files.
    """

    # Parse the XML
    tree = ET.parse(path)
    root = tree.getroot()
    file_name = Path(path).stem

    paths = []
    # Create new XML trees based on the split
    for idx, i in enumerate(range(0, len(root), max_children)):
        new_root = ET.Element(root.tag, root.attrib)
        new_root.extend(root[i:i + max_children])
        new_tree = ET.ElementTree(new_root)
        new_tree.write(f"{output_dir}/{file_name}_{idx}.xml")
        paths.append(f"{output_dir}/{file_name}_{idx}.xml")
    
    return paths
