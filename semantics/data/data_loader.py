from typing import Union, List, Optional, Literal
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
                 path: str
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
        # size = os.path.getsize(path)
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
                    image_dir: str,
                    ocr: Literal['easyocr', 'pytesseract'] = 'easyocr',
                    **kwargs
                    ):
        """
        Class method to read texts from images.
        
        Args:
            image_dir (str): Path to the directory containing the images.
            ocr (Literal['easyocr', 'pytesseract']): OCR engine to be used. Defaults to 'easyocr'.
            **kwargs: Keyword arguments to be passed to the ImageProcessor.preprocess() function.
        """
        if ocr == 'easyocr':
            ocr_engine = EasyOCR()
        elif ocr == 'pytesseract':
            ocr_engine = PytesseractOCR()
        else:
            raise ValueError("OCR engine not supported. Please use 'easyocr' or 'pytesseract'.")

        paths = sorted(glob.glob(f'{image_dir}/*.png'))
        preprocessing_options = kwargs.get('preprocessing_options', None)
        save_preprocessed = kwargs.get('save_preprocessed', False)
        if save_preprocessed:
            if not os.path.exists(kwargs.get('output_dir', f'{image_dir}/output')):
                os.makedirs(kwargs.get('output_dir', f'{image_dir}/output'))

        
        ocr_options = kwargs.get('ocr_options', None)
        save_as_txt = kwargs.get('save_as_txt', False)
        if save_as_txt:
            if not os.path.exists(kwargs.get('output_dir', f'{image_dir}/output')):
                os.makedirs(kwargs.get('output_dir', f'{image_dir}/output'))
            
        reader = ImageProcessor()
        texts = []

        for path in paths:
            print(f'Processing image: {Path(path).stem}')
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            preprocessed_image, patches = reader.preprocess(image=image, **preprocessing_options)
            del image
            if save_preprocessed:
                output_path = f"{kwargs.get('output_dir', f'{image_dir}/output')}/{Path(path).stem}.png"
                cv2.imwrite(output_path, preprocessed_image)


            if patches is not None:
                text = []
                for idx, patch in enumerate(patches):
                    output_path = f"{kwargs.get('output_dir', f'{image_dir}/output')}/{Path(path).stem}_{idx}.png"
                    cv2.imwrite(output_path, patch)

                    text += reader.ocr(image=patch, ocr=ocr_engine, **ocr_options)
                    del patch

            else:         
                text = reader.ocr(image=preprocessed_image, ocr=ocr_engine, **ocr_options)

            if save_as_txt:
                output_path = f"{kwargs.get('output_dir', f'{image_dir}/output')}/{Path(path).stem}.txt"
                with open(output_path, 'w') as f:
                    for line in text:
                        f.write(f'{line}\n')
    
            del preprocessed_image
            texts.append(text)
            del text

        return cls(texts)
    
    def sample(
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
            Original texts:  ['This is a test.', 'This is another test.', 'This is a third test.', 'This is also a third test.']
            >>> print('Filtered texts: ', Loader(texts).sample(target_words=['third'], max_documents=1, shuffle=False))
            Filtered texts:  ['This is a third test.']
            >>> print('Filtered texts: ', Loader(texts).sample(target_words='third', max_documents=1, shuffle=True, random_seed=42))
            Filtered texts:  ['This is also a third test.']
        """

        if target_words is not None:
            if isinstance(target_words, str):
                target_words = [target_words]

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



class EasyOCR():
    def __init__(self):
        import easyocr
        self.ocr = easyocr

    def forward(
            self,
            image: np.ndarray,
            language: str,
            detail: int = 0,
            **kwargs
        ):
        """
        Args:
            image (np.ndarray): Image to be processed.
            language (List[str]): List of languages to be used.
            detail (int, optional): Detail level. Defaults to 0.
            **kwargs: Keyword arguments to be passed to the easyocr.Reader.readtext() function.
            
        Returns:
            ocr_results (Union[List[str], Tuple[List[List[List[int]]], List[str]]]): List of texts or a tuple of coordinates and texts.
        
        Examples:
            >>> from semantics.data.data_loader import ImageProcessor
            >>> image = cv2.imread('input/images/test.png', cv2.IMREAD_GRAYSCALE)
            >>> print('Text: ', ImageProcessor().ocr(image=image, language='en'))
            Text:  ['This is a test.']
            >>> coordinates, texts = ImageProcessor().ocr(image=image, language='en', detail=1)
            >>> print('Coordinates: ', coordinates)
            >>> print('Texts: ', texts)
            Coordinates:  [[[0, 0], [0, 100], [100, 100], [100, 0]]]
            Texts:  ['This is a test.']
        
        """
        reader = self.ocr.Reader([language], gpu = False)
        result = reader.readtext(image, detail=detail, **kwargs)

        if detail == 0:
            return result
        
        coordinates, text_confidences = zip(*result)
        texts, _ = zip(*text_confidences)
        coordinates = list(coordinates)
        texts = list(texts)

        del text_confidences
        del result

        return coordinates, texts


class PytesseractOCR():
    def __init__(self):
        import pytesseract
        self.ocr = pytesseract
    
    def forward(
            self,
            image: np.ndarray,
            language: str,
            detail: int = 0,
            **kwargs
        ):
        
        if image.shape[-1] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        else:
            img_rgb = image

        if detail == 0:
            return self.ocr.image_to_string(img_rgb, lang=language, **kwargs)
        
        result = self.ocr.image_to_data(img_rgb, lang=language, output_type=self.ocr.Output.DICT, **kwargs)
        coordinates = list(zip(result['left'], result['top'], result['width'], result['height']))
        texts = result['text']
        del result
        return coordinates, texts



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
                angle = self._get_rotation_angle(image=copy_image)
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
        
        if kwargs.get('split', False):
            patches = self.split_image(image=copy_image)
            return copy_image, patches
            
        else:
            return copy_image, None

    def _get_rotation_angle(
            self,
            image: np.ndarray
        ) -> float:

        lines = self._lines(image=image)
        if len(lines) > 0:
            angles = []
            for line in lines:
                x1 = line[0]
                y1 = line[1]
                x2 = line[2]
                y2 = line[3]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                angles.append(angle)

            # Average out the angles for a rough estimate
            return np.mean(angles)
        else:
            return float(0)
      

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
            ocr: Union[EasyOCR, PytesseractOCR],
            **kwargs
        ):
        
        return ocr.forward(image=image, **kwargs)
        

    def _lines(
            self,
            image: np.ndarray
        ) -> List[List[int]]:
        copy_image = image.copy()
        lines = cv2.HoughLines(copy_image, 1, np.pi / 180, 200)
        all_lines = []
        if lines is not None:
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                all_lines.append([x1, y1, x2, y2])
        return all_lines

    def _linesP(
            self,
            image: np.ndarray
        ) -> List[List[int]]:
        copy_image = image.copy()
        lines = cv2.HoughLinesP(copy_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)

        all_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                all_lines.append([x1, y1, x2, y2])
        return all_lines
       
    
    def _draw_lines(
            self,
            image: np.ndarray,
            lines: List[List[int]]
        ):
        copy_image = image.copy()
        copy_image = cv2.cvtColor(copy_image,cv2.COLOR_GRAY2RGB)
        if len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(copy_image, (x1, y1), (x2, y2), (0, 255, 0), 15)
            
            return copy_image
    
        else:
            print('No lines detected.')
            return copy_image
        

    def _pageColumns(
            self,
            image: np.ndarray
        ) -> List[List[int]]:
        copy_image = image.copy()
        lines = self._linesP(image=copy_image)


        if len(lines) == 0:
            print('No columns detected.')
            return lines
        
        lines = sorted(lines, key=lambda x: x[0])
        all_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            if abs(x1 - x2) > 10 or abs(y1 - y2) < 500:
                continue

            if len(all_lines) == 0:
                all_lines.append(line)

            elif abs(x1 - all_lines[-1][0]) < 5:
                all_lines[-1][3] = image.shape[0]

            else:
                all_lines.append(line)
    
        clean_lines = []
        from sklearn.cluster import KMeans
        import numpy as np

        init = np.array([int(image.shape[1] / 6)* i for i in range(7)]).reshape(-1, 1)
        X = np.array([line[0] for line in all_lines]).reshape(-1, 1)

        clustering = KMeans(n_clusters=7, random_state=0, init=init).fit(X)
        labels = clustering.labels_
        centroid = clustering.cluster_centers_
        
        for c, label in enumerate(np.unique(labels)):
            x1 = np.mean([centroid[c][0].astype(int)] + [all_lines[i][0] for i in np.where(labels == label)[0]]).astype(int)
            y1 = 0
            x2 = x1
            y2 = image.shape[0]
            clean_lines.append([x1, y1, x2, y2])
        return clean_lines


    def _cut(
            self,
            image: np.ndarray,
            lines: List[List[int]]
        ) -> List[np.ndarray]:
        copy_image = image.copy()
        patches = []

        for i, line in enumerate(lines[:-1]):
            x1 = line[0]
            x2 = lines[i + 1][0]
            patch = copy_image[0:image.shape[0], x1:x2]
            patches.append(patch)
            
        return patches
    

    def split_image(
            self,
            image: np.ndarray
        ) -> List[np.ndarray]:
        copy_image = image.copy()
        lines = self._pageColumns(image=copy_image)
        patches = self._cut(image=copy_image, lines=lines)
        return patches
    




if __name__ == "__main__":
    # image_dir = 'input/images'
    # output_dir = image_dir + '/output'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    # preprocessing_options = {
        # 'rotation': True,
        # 'rotation_angle': None, # If None, the angle will be automatically determined. Otherwise, it should be a float value in degrees.
        # 'threshold': True,
        # 'threshold_val': (0, 255),
        # 'threshold_type': cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        # # 'blur': True,
        # 'blur_ksize': (5, 5),
        # 'blur_sigma': 0,
        # 'dilation': True,
        # 'dilation_kernel_shape': (5, 5),
        # 'dilation_iterations': 1,
        # 'erosion': True,
        # 'erosion_kernel_shape': (3, 3),
        # 'erosion_iterations': 2,
        # 'opening': True,
        # 'opening_kernel_shape': (5, 5),
        # 'closing': True,
        # 'closing_kernel_shape': (5, 5),
        # 'canny': True,
        # 'canny_thresh': (50, 150),
        # 'canny_L2gradient': False,
        # 'bitwise_not': True,
    #     'split': True,
    # }

    # ocr_model = 'easyocr'
    # ocr_options = {
    #     'language': 'en',
    #     'detail': 0,
    # }

    
    # data_loader = Loader.from_images(
    #     image_dir=image_dir,
    #     ocr=ocr_model,
    #     preprocessing_options=preprocessing_options,
    #     ocr_options=ocr_options,
    #     save_preprocessed=True,
    #     output_dir= f"{image_dir}/split",
    #     save_as_txt=True,
    # )

    # print(data_loader.texts)

    xml_dir = 'input/xml'
