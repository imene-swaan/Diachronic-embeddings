import xmltodict
import json
import os
from glob import glob
import re
import numpy as np


def read_xml(file_path: str):
        with open(file_path, 'r') as f:
            xml_string = f.read()
        return xmltodict.parse(xml_string)



def get_articles(file_path: str):
    """
    Read in the xml file and return a list of articles.

    Args:
        file_path (str): The path to the xml file.

    Returns:
        list: A list of articles.
    """
    xml_dict = read_xml(file_path)
    return xml_dict['records']['record']


def get_article_text(article: dict):
    """
    Get the text from an article.

    Args:
        article (dict): The article.

    Returns:
        str: The text of the article.
    """
    if 'fulltext' not in article.keys():
        return ''

    else:
        return article['fulltext']

 
def get_sentences(records: list):
    """
    Get all sentences from a list of articles (dict).

    Args:
        records (list): A list of articles (dict).

    Returns:
        list: A list of sentences.
    """
    sentences = []
    for record in records:
        article = get_article_text(record)

        if article == '':
            continue
        
        else:
            sentences.extend(article.split('. '))
        
    return sentences


def save_sentences(sentences: list, file_path: str):
    """
    Save a list of sentences to a file.

    Args:
        sentences (list): A list of sentences.
        file_path (str): The path to the file.
    """
    with open(file_path, 'w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')


# ------------------- cleantxt --------------------
def cleantxt(text, remove_punctuation, lowercase, lemmatize, remove_stopwords):
    """
    Simple text preprocessing function. Define your args in the config.yaml file.

    Args:
        text (str): The text to be preprocessed.
        remove_punctuation (bool): Whether punctuation should be removed.
        lowercase (bool): Whether text should be lowercased.

        remove_stopwords (bool): Whether stopwords should be removed.

    Returns:
        str: Preprocessed text (without punctuation/lowercased depending on args).
    """
    newtext = re.sub('-\n', '', text) # Remove OCR'd linebreaks within words if they exist
    newtext = re.sub('\n', ' ', newtext) # Remove ordinary linebreaks (there shouldn't be, so this might be redundant)
    if remove_punctuation == True:
        newtext = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', str(newtext)) # Remove anything that is not a space, a letter, a dot, or a number
    if lowercase == True:
        newtext = str(newtext).lower() # Lowercase
    
    if lemmatize == True:
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet 

        lemmatizer = WordNetLemmatizer()

        newtext = ' '.join(list(map(lambda x: lemmatizer.lemmatize(x, wordnet.VERB), newtext.split())))
    
    if remove_stopwords == True:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        newtext = ' '.join(list(filter(lambda x: x not in stop_words, newtext.split())))

    return newtext











if __name__ == '__main__':
    path = '../../data/articles_raw_data/'
    file_path = glob(os.path.join(path, '*.xml'))[0]

    print(len(read_xml(file_path)['records']['record']))
    