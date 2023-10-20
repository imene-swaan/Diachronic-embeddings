from typing import List, Union
from pathlib import Path
from gensim.models import Word2Vec
from src.utils.utils import smart_procrustes_align_gensim
import os
from pathlib import Path



class Word2VecTrainer:
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
    
    def __init__(
            self,
            model_path: str = None,
            min_count=5,
            window=5,
            negative=5,
            ns_exponent=0.75,
            vector_size=300,
            workers=1,
            sg=1,
            **kwargs
            ):
        
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
            output_dir: Union[str, Path] = None,
            epochs=5,
            start_alpha=0.025,
            end_alpha=0.0001,
            compute_loss=True,
            **kwargs
            ):
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
        self.model_paths = model_paths
        self.reference_model = None
        self.models = []
        self.model_names = [Path(model_path).stem for model_path in model_paths]
        self.aligned_models = []

        self.load_models()

    def load_models(self):
        for model_path in self.model_paths:
            self.models.append(Word2Vec.load(model_path))

    def align_models(
            self,
            reference_index: int = -1,
            output_dir: str = None,
            method: str = "procrustes",
            ):
        
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




class WordEmbeddings:
    def __init__(
            self,
            pretrained_model_path: str,
            ):
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
        self.model = Word2Vec.load(self.model_path)
        self.vocab = True
    
    def infer_vector(self, word:str, norm = False) -> List[float]:
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        return self.model.wv.get_vector(word, norm = norm)
    
    


class Word2VecInference:
    def __init__(
            self,
            pretrained_model_path: str,
            ):
        self.word_vectorizor = WordEmbeddings(pretrained_model_path)
    
    def get_embedding(self, word:str, norm: bool = False):
        return self.word_vectorizor.infer_vector(word= word, norm = norm)
    
    def get_similarity(self, word1: str, word2: str):
        return self.word_vectorizor.model.wv.similarity(word1, word2)
    
    def get_top_k_words(
            self,
            word: str,
            k: int = 10,
            ):

        try:
            sims = self.word_vectorizor.model.wv.most_similar(
                word,
                topn=k
                )
            return tuple(map(list, zip(*sims)))
        
        except KeyError:
            print("The word in the input is not in the model vocabulary.")
            return []
        


if __name__ == "__main__":
    pass
