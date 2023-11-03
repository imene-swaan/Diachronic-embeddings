from typing import List, Union
from pathlib import Path
from gensim.models import Word2Vec
from semantics.utils.utils import smart_procrustes_align_gensim
import os
from pathlib import Path
from typing import List, Optional, Union



class Word2VecTrainer:    
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
        Wrapper class for gensim.models.Word2Vec
        
        Parameters
        ----------
        model_path : str, optional
            Path to a pretrained model, by default None
            min_count : int, optional
            window : int, optional
            negative : int, optional
            ns_exponent : float, optional
            vector_size : int, optional
            workers : int, optional
            sg : int, optional
            **kwargs : optional
                Additional parameters for gensim.models.Word2Vec

        Attributes
        ----------
        model : gensim.models.Word2Vec
            The Word2Vec model

        Methods
        -------
        train(data, output_path, epochs, alpha, min_alpha, compute_loss, **kwargs)
            Train the Word2Vec model on the given data

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
        
        Parameters
        ----------
        data : List[str]
            List of documents
        output_path : Union[str, Path], optional
            Path to save the trained model, by default None
        epochs : int, optional
            Number of epochs, by default 5
        start_alpha : float, optional
            Learning rate, by default 0.025
        end_alpha : float, optional
            Minimum learning rate, by default 0.0001
        compute_loss : bool, optional
            Whether to compute the loss, by default True
        **kwargs : optional
            Additional parameters for gensim.models.Word2Vec.train
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
    def __init__(
            self, 
            model_paths: List[str],
            
            ):
        """
        Aligns multiple Word2Vec models.
        
        Parameters
        ----------
        model_paths : List[str]
            List of paths to the models
        
        Attributes
        ----------
        model_paths : List[str]
            List of paths to the models
        reference_model : gensim.models.Word2Vec
            The reference model
        models : List[gensim.models.Word2Vec]
            List of models
        model_names : List[str]
            List of model names
        aligned_models : List[gensim.models.Word2Vec]
            List of aligned models
            
        Methods
        -------
        load_models()
            Load the models
        align_models(reference_index, output_dir, method)
            Align the models        
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

        Parameters
        ----------
        reference_index : int, optional
            Index of the reference model, by default -1
        output_dir : str, optional
            Path to save the aligned models, by default None
        method : str, optional
            Alignment method, by default "procrustes"

        Returns
        -------
        List[Word2Vec]
            List of aligned models
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
    def __init__(
            self,
            pretrained_model_path: Optional[str] = None,
            ):
        """
        Wrapper class for gensim.models.Word2Vec
        
        Parameters
        ----------
        pretrained_model_path : str
            Path to a pretrained model, by default None

        Attributes
        ----------
        model_path : str
            Path to the pretrained model
        model : gensim.models.Word2Vec
            The Word2Vec model
        vocab : bool
            Whether the model has been initialized

        Methods
        -------
        _word2vec_case_preparation()
            Prepare the Word2Vec model
        infer_vector(word, norm)
            Infer the vector of a word
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
        Prepare the Word2Vec model
        """
        if self.model_path is None:
            self.model = Word2Vec()
        else:
            self.model = Word2Vec.load(self.model_path)
        self.vocab = True
    
    def infer_vector(self, word:str, norm = False) -> List[float]:
        """
        Infer the vector of a word

        Parameters
        ----------
        word : str
            The word
        norm : bool, optional
            Whether to normalize the vector, by default False

        Returns
        -------
        List[float]
            The vector of the word
        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        return self.model.wv.get_vector(word, norm = norm)
    
    


class Word2VecInference:
    def __init__(
            self,
            pretrained_model_path: Optional[str] = None,
            ):
        """
        Wrapper class for gensim.models.Word2Vec

        Parameters
        ----------
        pretrained_model_path : str
            Path to a pretrained model, by default None

        Attributes
        ----------
        word_vectorizor : WordEmbeddings
            The Word2Vec model

        Methods
        -------
        get_embedding(word, norm)
            Infer the vector of a word
        get_similarity(word1, word2)
            Get the cosine similarity between two words
        get_top_k_words(word, k)
            Get the top k most similar words to a word in the vocabulary of the model. Default k = 10  
        """
        self.word_vectorizor = Word2VecEmbeddings(pretrained_model_path)
    
    def get_embedding(self, word:str, norm: bool = False) -> List[float]:
        """
        Infer the vector of a word
        
        Parameters
        ----------
        word : str
            The word
        norm : bool, optional
            Whether to normalize the vector, by default False

        Returns
        -------
        List[float]
            The vector of the word
        """
        return self.word_vectorizor.infer_vector(word= word, norm = norm)
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Get the cosine similarity between two words

        Parameters
        ----------
        word1 : str
            The first word
        word2 : str
            The second word

        Returns
        -------
        float
            The cosine similarity between the two words
        """
        return self.word_vectorizor.model.wv.similarity(word1, word2)
    
    def get_top_k_words(
            self,
            word: str,
            k: int = 10,
            ):
        """
        Get the top k most similar words to a word in the vocabulary of the model. Default k = 10

        Parameters
        ----------
        word : str
            The word
        k : int, optional
            The number of similar words to return, by default 10

        Returns
        -------
        List[str]
            List of similar words
        """

        try:
            sims = self.word_vectorizor.model.wv.most_similar(
                word,
                topn=k
                )
            return tuple(map(list, zip(*sims)))
        
        except KeyError:
            print("The word in the input is not in the model vocabulary.")
            return [], []
        


if __name__ == "__main__":
    pass
