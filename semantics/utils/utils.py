import yaml
import tomli
import random
import numpy as np
from typing import List, Union, Optional, Tuple
from collections import Counter
import colorsys


def generate_colors(num_colors=3, range_min=0.0, range_max=1.0):
    """
    Generate random pastel colors.

    Args:
        num_colors (int): The number of colors to generate.
    
    Returns:
        colors (List[str]): A list of colors in hex format.
    """
    colors = []
    for i in range(num_colors):
        # Evenly distribute hue across the spectrum
        hue = i / num_colors
        # Set saturation and lightness to high values for shiny colors
        saturation = 0.9  # High saturation for vivid color
        lightness = 0.5   # Balanced lightness for shininess
        
        # Convert HSL to RGB
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        
        # Convert RGB from 0-1 range to 0-255 range and format as hex
        color = '#{0:02x}{1:02x}{2:02x}'.format(int(r*255), int(g*255), int(b*255))
        colors.append(color)
    
    return colors

    # ['#{:02x}{:02x}{:02x}'.format(
    #     int(random.uniform(range_min, range_max) * 255),
    #     int(random.uniform(range_min, range_max) * 255),
    #     int(random.uniform(range_min, range_max) * 255)
    # ) for _ in range(num_colors)]




def read_toml(config_path: str) -> dict:
    """
    Read in a config file and return a dictionary.

    Args:
        config_path (str): The path to the config file.

    Returns:
        dict: The dictionary.
    """
    with open(config_path, "rb") as f:
        return tomli.load(f)

def read_yaml(file_path: str) -> dict:
    """
    Read in a yaml file and return a dictionary.

    Args:
        file_path (str): The path to the yaml file.

    Returns:
        dict: The dictionary.
    """
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def read_txt(file_path: str) -> list:
    """
    Read in a txt file and return a list of lines.

    Args:
        file_path (str): The path to the txt file.

    Returns:
        list: A list of lines.
    """

    with open(file_path) as f:
        return f.readlines()
    

def train_test_split(data: List[str], test_ratio=0.2, random_seed=None):
    """
    Split the data into train and test sets.

    Args:
        data (List[str]): The data to split.
        test_ratio (float): The ratio of the test set.
        random_seed (int): The random seed.

    Returns:
        data Tuple[List[str], List[str]]: The training and testing datasets.
    """
    
    if random_seed:
        random.seed(random_seed)
    data_copy = data[:]
    random.shuffle(data_copy)
    split_idx = int(len(data_copy) * (1 - test_ratio))
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]
    return train_data, test_data


def sample_data(data: list, sample_size: int, random_seed=None):
    """
    Sample data.

    Args:
        data (list): The data to sample.
        sample_size (int): The size of the sample.
        random_seed (int): The random seed.

    Returns:
        sample_data (list): The sampled data.
    """
    
    if random_seed:
        random.seed(random_seed)
    data_copy = data[:]
    random.shuffle(data_copy)
    return data_copy[:sample_size]


def count_occurence(data: List[str], word: Optional[Union[List[str], str]] = None) -> int:
    """
    Count the occurence of one, multiple, or all words in a list of strings (data).

    Args:
        data (List[str]): The list of strings.
        word (Union[List[str], str]): The word or list of words to count. If word is str, the function will count the occurence of the word in each data string. If a list of words is provided, the function will count the co-occurence of the words in each data string. If word is None, the function return the count of all words in the data. Defaults to None.

    Returns:
        count (int): The count of occurences.
    """

    if word is None:
        return sum([len(sentence.split()) for sentence in data])

    if isinstance(word, str):
        word = [word]

    if len(word) == 1:
        return sum([1 for sentence in data if word[0] in sentence])
    
    else:
        counts = 0
        for sentence in data:
            if all([w in sentence for w in word]):
                counts += 1
        return counts
            
        
 
def most_frequent(my_list: List[str], n: int = 1) -> Tuple[List[str], List[int]]:
    """
    Return the n most frequent words in a list.
    
    Args:
        my_list (List[str]): The list of words.
        n (int): The number of most frequent words to return.
        
    Returns:
        most_frequent ((List[str], List[int])): The n most frequent words and their counts
    """
    occurence_count = Counter(my_list)
    most_common = occurence_count.most_common(n)
    return [w for w, _ in most_common], [c for _, c in most_common]


# ------------------- smart_procrustes_align_gensim --------------------
# from: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.

    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # re-filling the normed vectors
    in_base_embed.wv.fill_norms(force=True)
    in_other_embed.wv.fill_norms(force=True)

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
    return other_embed

# ------------------- intersection_align_gensim --------------------
# from: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)
