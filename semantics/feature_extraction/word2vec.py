from typing import List, Union
from pathlib import Path
from gensim.models import Word2Vec
from semantics.utils.utils import smart_procrustes_align_gensim
import os
from pathlib import Path
from typing import List, Optional, Union
import re
from nltk.corpus import stopwords

class Word2VecTrainer:  
    """
    Wrapper class for gensim.models.Word2Vec to train a Word2Vec model.

    Methods
    -------
        __init__(model_path, min_count, window, negative, ns_exponent, vector_size, workers, sg, **kwargs)
            Initialize the Word2Vec model
        train(data, output_path, epochs, start_alpha, end_alpha, compute_loss, **kwargs)
            Train the Word2Vec model on the given data
    """  

    def __init__(
            self,
            model_path: Optional[str] = None,
            min_count=0,
            window=15,
            negative=5,
            ns_exponent=0.75,
            vector_size=100,
            workers=1,
            sg=1,
            **kwargs
            ):
        """
        Args:
            model_path (str, optional): Path to a pretrained model, by default None.
            min_count (int, optional): Ignores all words with total frequency lower than this, by default 0
            window (int, optional): The maximum distance between the current and predicted word within a sentence, by default 15
            negative (int, optional): If > 0, negative sampling will be used, by default 5
            ns_exponent (float, optional): The exponent used to shape the negative sampling distribution, by default 0.75
            vector_size (int, optional): Dimensionality of the word vectors, by default 100
            workers (int, optional): Number of worker threads to train the model, by default 1
            sg (int, optional): Training algorithm: 1 for skip-gram; otherwise CBOW, by default 1
            **kwargs (optional): Additional arguments to pass to the gensim.models.Word2Vec constructor

        Attributes:
            model (gensim.models.Word2Vec): The Word2Vec model
        """
        
        if model_path:
            self.model = Word2Vec.load(model_path)
        else:
            self.model = Word2Vec(
                    min_count=min_count,
                    window=window,
                    negative=negative,
                    ns_exponent=ns_exponent,
                    vector_size=vector_size,
                    workers=workers,
                    sg=sg,
                    **kwargs
                    )
        
    def train(
            self, 
            data: List[str],
            output_dir: Optional[Union[str, Path]] = None,
            epochs=5,
            start_alpha=0.025,
            end_alpha=0.0001,
            compute_loss=True,
            **kwargs
            ):
        """
        Train the Word2Vec model on the given data
        
        Args:
            data (List[str]): List of documents
            output_dir (Union[str, Path], None): Path to save the trained model, by default None
            epochs (int, optional): Number of epochs, by default 5
            start_alpha (float, optional): Learning rate, by default 0.025
            end_alpha (float, optional): Minimum learning rate, by default 0.0001
            compute_loss (bool, optional): Whether to compute the loss, by default True
            **kwargs : optional

        Examples:
            >>> from semantics.feature_extraction.word2vec import Word2VecTrainer
            >>> texts = ['This is a test.', 'This is another test.', 'This is a third test.']
            >>> Word2VecTrainer().train(texts, epochs=1)
            >>> print('Trained model: ', Word2VecTrainer().model)
            Trained model:  Word2Vec(vocab=5, vector_size=100, alpha=0.025)
        """
        self.model.build_vocab(data)
        total_examples = self.model.corpus_count
        self.model.train(
                data,
                total_examples=total_examples,
                epochs=epochs,
                start_alpha=start_alpha,
                end_alpha=end_alpha,
                compute_loss=compute_loss,
                **kwargs
                )
        if output_dir:
            self.model.save(output_dir)


class Word2VecAlign:
    """
    Wrapper class for gensim.models.Word2Vec to align Word2Vec models.

    Methods
    -------
        __init__(model_paths)
            Initialize the Word2VecAlign object with a list of paths to the Word2Vec models.
        load_models()
            Load the models
        align_models(reference_index, output_dir, method)
            Align the models
    """
    def __init__(
            self, 
            model_paths: List[str],
            
            ):
        """
        Args:
            model_paths (List[str]): List of paths to the models 

        Attributes:
            model_paths (List[str]): List of paths to the models 
            reference_model (gensim.models.Word2Vec): The reference model
            models (List[gensim.models.Word2Vec]): List of models
            model_names (List[str]): List of model names
            aligned_models (List[gensim.models.Word2Vec]): List of aligned models     
        """
        self.model_paths = model_paths
        self.reference_model = None
        self.models = []
        self.model_names = [Path(model_path).stem for model_path in model_paths]
        self.aligned_models = []

        self.load_models()

    def load_models(self) -> None:
        """
        Load the models
        """
        for model_path in self.model_paths:
            self.models.append(Word2Vec.load(model_path))

    def align_models(
            self,
            reference_index: int = -1,
            output_dir: Optional[str] = None,
            method: str = "procrustes",
            ) -> List[Word2Vec]:
        """
        Align the models

        Args: 
            reference_index (int, optional): Index of the reference model, by default -1
            output_dir (str, optional): Path to save the aligned models, by default None
            method (str, optional): Alignment method, by default "procrustes"
      
        Returns:
            aligned_models (List[gensim.models.Word2Vec]): List of aligned models

        Examples:
            >>> from semantics.feature_extraction.word2vec import Word2VecAlign
            >>> model_paths = ['model1.model', 'model2.model']
            >>> Word2VecAlign(model_paths).align_models(reference_index=0, output_dir='aligned_models')
            >>> print('Aligned models: ', Word2VecAlign(model_paths).aligned_models)
            Aligned models:  [Word2Vec(vocab=5, vector_size=100, alpha=0.025), Word2Vec(vocab=5, vector_size=100, alpha=0.025)]
        """
        
        if method != "procrustes":
            raise NotImplementedError("Only procrustes alignment is implemented. Please use method='procrustes'")

        
        self.reference_model = self.models[reference_index]
        self.reference_model.save(f"{output_dir}/{self.model_names[reference_index]}_aligned.model")
        self.aligned_models.append(self.reference_model)
        self.models.pop(reference_index)

        for i, model in enumerate(self.models):
            aligned_model = smart_procrustes_align_gensim(self.reference_model,model)
            aligned_model.save(f"{output_dir}/{self.model_names[i]}_aligned.model")
            self.aligned_models.append(aligned_model)

        return self.aligned_models




class Word2VecEmbeddings:
    """
    Wrapper class for gensim.models.Word2Vec to infer word vectors.

    Methods
    -------
        __init__(pretrained_model_path)
            Initialize the Word2VecEmbeddings object with a pretrained model.
        _word2vec_case_preparation()
            Prepare the Word2Vec model
        infer_vector(word, norm)
            Infer the vector of a word
    """
    def __init__(
            self,
            pretrained_model_path: Optional[str] = None,
            ):
        """
        Args: 
            pretrained_model_path (str, optional): Path to a pretrained model, by default None
        
        Attributes:
            model_path (str, optional): Path to a pretrained model, by default None
            model (gensim.models.Word2Vec): The Word2Vec model
            vocab (bool): Whether the model has been initialized
        """
        self.model_path = pretrained_model_path
        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise ValueError(
                    f"Model path {pretrained_model_path} does not exist."
                )
            self.model_path = pretrained_model_path
        
        self.model = None
        self.vocab = False

        self._word2vec_case_preparation()
    
    def _word2vec_case_preparation(self) -> None:
        """
        Initialize the Word2Vec model
        """
        if self.model_path is None:
            self.model = Word2Vec()
        else:
            self.model = Word2Vec.load(self.model_path)
        self.vocab = True
    
    def infer_vector(self, word:str, norm = False) -> List[float]:
        """
        Infer the vector of a word

        Args:
            word (str): The word to infer the embedding vector of
            norm (bool, optional): Whether to normalize the vector, by default False

        Returns:
            embedding (List[float]): The embedding vector of the word
        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        return self.model.wv.get_vector(word, norm = norm)
    
    


class Word2VecInference:
    """
    Wrapper class for gensim.models.Word2Vec for Inference.

    Methods
    -------
        __init__(pretrained_model_path)
            Initialize the Word2VecInference object with a pretrained model.
        get_embedding(word, norm)
            Infer the vector of a word
        get_similarity(word1, word2)
            Get the cosine similarity between two words
        get_top_k_words(word, k)
            Get the top k most similar words to a word in the vocabulary of the model.
    """
    def __init__(
            self,
            pretrained_model_path: Optional[str] = None,
            ):
        """
        Args:
            pretrained_model_path (str, optional): Path to a pretrained model, by default None  

        Attributes:
            word_vectorizor (Word2VecEmbeddings): The Word2VecEmbeddings object
        """
        self.word_vectorizor = Word2VecEmbeddings(pretrained_model_path)
    
    def get_embedding(self, word:str, norm: bool = False) -> List[float]:
        """
        Infer the vector of a word
        
        Args:
            word (str): The word to infer the embedding vector of
            norm (bool, optional): Whether to normalize the vector, by default False
        
        Returns:
            embedding (List[float]): The embedding vector of the word

        Examples:
            >>> from semantics.feature_extraction.word2vec import Word2VecInference
            >>> Word2VecInference('model.model').get_embedding('test', norm=False)
            array([-0.00460768, -0.00460768, ..., -0.00460768, -0.00460768])
        """
        return self.word_vectorizor.infer_vector(word= word, norm = norm)
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Get the cosine similarity between two words' embedding vectors

        Args:
            word1 (str): The first word
            word2 (str): The second word
        
        Returns:
            similarity (float): The cosine similarity between the two words
        
        Examples:
            >>> from semantics.feature_extraction.word2vec import Word2VecInference
            >>> Word2VecInference('model.model').get_similarity('test', 'another')
            0.99999994
        """
        return self.word_vectorizor.model.wv.similarity(word1, word2)
    
    def get_top_k_words(
            self,
            main_word: str,
            k: int = 10,
            ):
        """
        Get the top k most similar words to a word in the vocabulary of the model. Default k = 10

        Args:
            main_word (str): The word to get the top k most similar words of
            k (int, optional): The number of words to return, by default 10
        
        Returns:
            topk (Tuple[List[str], List[float]]): Tuple of lists of the top k most similar words and their cosine similarities
        
        Examples:
            >>> from semantics.feature_extraction.word2vec import Word2VecInference
            >>> Word2VecInference('model.model').get_top_k_words('test', k=1)
            (['another'], [0.9999999403953552])
        """

        try:
            sims = self.word_vectorizor.model.wv.most_similar(
                main_word,
                topn=k
                )
            
            words, _= tuple(map(list, zip(*sims)))

            top_k_words = list(map(lambda x: re.sub(r"\W", '', x), words))
            stop_words = list(set(stopwords.words('english')))
            top_k_words = list(filter(lambda x: all([x != main_word, x not in main_word, main_word not in x, len(x) > 2, x not in stop_words]), top_k_words))
            return top_k_words
        
        except KeyError:
            print(f"The word {main_word} in the input is not in the model vocabulary.")
            return []
        


if __name__ == "__main__":
    pass
