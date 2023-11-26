
from typing import Union, List, Optional
from pathlib import Path
import xml.etree.ElementTree as ET 
from semantics.utils.utils import read_txt, sample_data
import os
import glob
import cv2
import numpy as np

class Loader():
    """
    Class for loading data.

    Methods
    -------
        from_txt: Reads texts from a text file.
        from_xml: Reads texts from an XML file.
        forward: Filters the texts based on the target words and the maximum number of documents.
    """
    def __init__(
            self,
            texts: List[str]
            ):
        """
        Args:
            texts (List[str]): List of texts.
        
        Attributes:
            texts (List[str]): List of texts.
        """
        
        self.texts = texts
    

    @classmethod
    def from_txt(cls,
                 path: Union[str, Path]
                 ):
        """
        Class method to read texts from a text file.

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
        Class method to read texts from an XML file.
        
        Args:
            path (Union[str, Path]): Path to the XML file.
            tag (str): Tag of the XML file to extract the texts from.
        """
        size = os.path.getsize(path)
        # if size > 1e8:
        #     raise ValueError("File size is too large. Please split the file into smaller files.")
        tree = ET.parse(path)
        root = tree.getroot()
        texts = []
        for elem in root.findall('.//' + tag):
            if isinstance(elem.text, str):
                texts.append(elem.text)
        return cls(texts)
    
    @classmethod
    def from_images(cls,
                    image_dir: Union[str, Path],
                    **kwargs
                    ):
        """
        Class method to read texts from images.
        
        Args:
            path (Union[str, Path]): Path to the directory containing the images.
        """
        paths = glob.glob(f'{image_dir}/*.png')
        preprocessing_options = kwargs.get('preprocessing_options', None)
        ocr_options = kwargs.get('ocr_options', None)
        reader = ImageProcessor()
        texts = []

        for path in paths:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            preprocessed_image = reader.preprocess(image=image, **preprocessing_options)
            _, text = reader.ocr(image=preprocessed_image, **ocr_options)
            texts.append(text)

        return cls(texts)
    
    def forward(
            self, 
            target_words: Optional[Union[List[str], str]] = None, 
            max_documents: Optional[int] = None, 
            shuffle: bool = True, 
            random_seed: Optional[int] = None
            ) -> List[str]:
        """
        Filters the texts based on the target words and the maximum number of documents.

        Args:
            target_words (List[str], str, None): List of target words. Defaults to None.
            max_documents (int, None): Maximum number of documents. Defaults to None.
            shuffle (bool): Whether to shuffle the data. Defaults to True.
            random_seed (int, None): Random seed. Defaults to None.

        Returns:
            texts (List[str]): List of texts.
        
        
        Examples:
            >>> from semantics.data.data_loader import Loader
            >>> texts = ['This is a test.', 'This is another test.', 'This is a third test.']
            >>> print('Original texts: ', texts)
            >>> print('Filtered texts: ', Loader(texts).forward(target_words=['third'], max_documents=1, shuffle=False))
            Original texts:  ['This is a test.', 'This is another test.', 'This is a third test.
            Filtered texts:  ['This is a third test.']
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
        paths (List[str]): List of paths to the new XML files.
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






class ImageProcessor():
    def __init__(
        self
        ):
        pass
    
    def preprocess(
            self,
            image: np.ndarray,
            **kwargs
        ):

        copy_image = image.copy()
        if kwargs.get('rotation', False):
            angle = kwargs.get('rotation_angle', None)
            if angle is None:
                angle = self._get_rotation_angle(grey_image=copy_image)
            if angle != 0:
                copy_image = self._rotate_image(image=copy_image, angle=angle)
    
        if kwargs.get('threshold', False):
            val = kwargs.get('threshold_val', (0, 255))
            thresh_type = kwargs.get('threshold_type', cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            copy_image = cv2.threshold(copy_image, val[0], val[1], type= thresh_type)[1]
        
        if kwargs.get('blur', False):
            val = kwargs.get('blur_ksize', (5, 5))
            sigma = kwargs.get('blur_sigma', 0)
            copy_image = cv2.GaussianBlur(copy_image, val, sigma)

        if kwargs.get('dilation', False):
            val = kwargs.get('dilation_kernel_shape', (5, 5))
            i = kwargs.get('dilation_iterations', 1)
            kernel = np.ones(val, np.uint8)
            copy_image = cv2.dilate(copy_image, kernel, iterations=i)
        
        if kwargs.get('erosion', False):
            val = kwargs.get('erosion_kernel_shape', (5, 5))
            i = kwargs.get('erosion_iterations', 1)
            kernel = np.ones(val, np.uint8)
            copy_image = cv2.erode(copy_image, kernel, iterations=i)
        
        if kwargs.get('opening', False):
            val = kwargs.get('opening_kernel_shape', (5, 5))
            kernel = np.ones(val, np.uint8)
            copy_image = cv2.morphologyEx(copy_image, cv2.MORPH_OPEN, kernel)
        

        if kwargs.get('closing', False):
            val = kwargs.get('closing_kernel_shape', (5, 5))
            kernel = np.ones(val, np.uint8)
            copy_image = cv2.morphologyEx(copy_image, cv2.MORPH_CLOSE, kernel)

        if kwargs.get('canny', False):
            val = kwargs.get('canny_thresh', (50, 150))
            l2 = kwargs.get('canny_L2gradient', False)
            copy_image = cv2.Canny(copy_image, val[0], val[1], apertureSize=3, L2gradient=l2)
        
        if kwargs.get('bitwise_not', False):
            copy_image = cv2.bitwise_not(copy_image)
            
        return copy_image

    def _get_rotation_angle(
            self,
            grey_image: np.ndarray
        ):
        edges = cv2.Canny(grey_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            angles = []
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                angles.append(angle)

            # Average out the angles for a rough estimate
            return np.mean(angles)
        return 0  # No lines detected

    def _rotate_image(
            self,
            image: np.ndarray,
            angle: float
        ):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


    def ocr(
            self,
            image: np.ndarray,
            **kwargs
        ):
        
        cope_image = image.copy()
        model = kwargs.get('model', 'easyocr')
        language = kwargs.get('language', 'en')
        if model == 'easyocr':
            return self._easyocr(image=cope_image, language=language)
        elif model == 'pytesseract':
            return self._pytesseract(image=cope_image, language=language)
        else:
            raise ValueError("Invalid model. Please choose from 'easyocr' or 'pytesseract'.")


    def _easyocr(
            self,
            image: np.ndarray,
            language: List[str],
            detail: int = 0,
            **kwargs
        ):
        
        import easyocr
        reader = easyocr.Reader(language, gpu = False)
        result = reader.readtext(image, detail=detail, **kwargs)

        if detail == 0:
            return result
        
        # Extracting the individual lists using map and zip
        coordinates, text_confidences = zip(*result)
        texts, _ = zip(*text_confidences)
        # Converting tuples back to lists (if needed)
        coordinates = list(coordinates)
        texts = list(texts)
        # confidences = list(confidences)
        del text_confidences
        del result
        return coordinates, texts


    
    def _pytesseract(
            self,
            image: np.ndarray,
            language: str,
            detail: int = 0,
            **kwargs
        ):
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        import pytesseract

        if detail == 0:
            return pytesseract.image_to_string(img_rgb, lang=language, **kwargs)
        
        result = pytesseract.image_to_data(img_rgb, lang=language, output_type=pytesseract.Output.DICT, **kwargs)
        coordinates = result['left'], result['top'], result['width'], result['height']
        texts = result['text']
        del result
        return coordinates, texts






if __name__ == "__main__":
    