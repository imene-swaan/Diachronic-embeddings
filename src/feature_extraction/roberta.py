import os
import logging
import tqdm
from typing import Union, List, Optional
from pathlib import Path
import math
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup, logging as lg
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from accelerate import Accelerator
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))) # Add src to path
from src.utils.utils import train_test_split






logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """
    This class is used to create a custom dataset for the Roberta model.
    
    Methods
    -------
        __init__(data: List[str], tokenizer, max_length=128, truncation=True, padding=True)
            The constructor for the CustomDataset class.
        __len__()
            This method is used to get the length of the dataset.
        __getitem__(idx)
            This method is used to get the item at a specific index.  
    """
    def __init__(
            self, 
            data: List[str], 
            tokenizer, 
            max_length=128,
            truncation=True,
            padding= "max_length",
            ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_data = tokenizer(data, truncation=truncation, padding=padding, max_length=max_length)

    def __len__(self):
        return len(self.tokenized_data.input_ids)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index.

        Parameters:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing the input_ids, attention_mask, and labels.
        """
        # Get the tokenized inputs at the specified index
        input_ids = self.tokenized_data.input_ids[idx]
        attention_mask = self.tokenized_data.attention_mask[idx]

        # Return a dictionary containing input_ids, attention_mask, and labels (if applicable)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
            # Add 'labels': labels if you have labels for your data
        }

class RobertaTrainer:
    """
    This class is used to train a Roberta model.

    Methods
    -------
        __init__(model_name="roberta-base", max_length=128, mlm_probability=0.15, batch_size=4, learning_rate=1e-5, epochs=3, warmup_steps=500, split_ratio=0.8)
            The constructor for the RobertaTrainer class.
        prepare_dataset(data)
            This method is used to prepare the dataset for training.
        train(data, output_dir: Union[str, Path] = None)
            This method is used to train the model.
    """
    def __init__(
            self, 
            model_name: str = "roberta-base", 
            max_length: int = 128, 
            mlm_probability: float = 0.15, 
            batch_size: int = 4, 
            learning_rate: float = 1e-5, 
            epochs: int = 3, 
            warmup_steps: int = 500, 
            split_ratio: float = 0.8, 
            truncation: bool = True, 
            padding: str = "max_length"
            ):

        """
        Args:
            model_name (str): Name of the model to train. Defaults to "roberta-base".
            max_length (int): Maximum length of the input sequence. Defaults to 128.
            mlm_probability (float): Probability of masking tokens in the input sequence. Defaults to 0.15.
            batch_size (int): Size of the batch. Defaults to 4.
            learning_rate (float): Learning rate of the optimizer. Defaults to 1e-5.
            epochs (int): Number of epochs to train the model for. Defaults to 3.
            warmup_steps (int): Number of warmup steps for the learning rate scheduler. Defaults to 500.
            split_ratio (float): Ratio to split the data into train and test. Defaults to 0.8.
            truncation (bool): Whether to truncate the input sequence to max_length or not. Defaults to True.
            padding (str): Whether to pad the input sequence to max_length or not. Defaults to "max_length".
        """
        
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=mlm_probability
            )

        self.split_ratio = split_ratio
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.accelerator = Accelerator()

    def prepare_dataset(self, data: List[str]):
        """
        This method is used to prepare the dataset for training.
        Args:
            data: List of strings to train the model on.
            
        Returns:
            train_loader: DataLoader object containing the training data.
            dataset: CustomDataset object containing the training data.
        """
        dataset = CustomDataset(
            data, 
            self.tokenizer, 
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding
            )
        
        train_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.data_collator
            )
        
        return train_loader, dataset

    def train(
            self, 
            data: List[str],
            output_dir: Optional[Union[str, Path]] = None
            ):
        """
        This method is used to train the model.
        Args:
            data (List[str]): List of strings to train the model on.
            output_dir (str, Path, None): Path to save the model to. Defaults to None.

        Returns:
            None
        """
        
        train_data, test_data = train_test_split(
            data, 
            test_ratio=1 - self.split_ratio, 
            random_seed=42
            )
        
        train_loader, _ = self.prepare_dataset(train_data)
        test_loader, _ = self.prepare_dataset(test_data)
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate
            )

        model, optimizer, train_loader, test_loader = self.accelerator.prepare(
            self.model, 
            optimizer, 
            train_loader, 
            test_loader
            )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warmup_steps, 
            num_training_steps=len(train_loader) * self.epochs
            )

        progress_bar = tqdm.tqdm(
            range(len(train_loader) * self.epochs), 
            desc="Training", 
            dynamic_ncols=True
            )
        
        for epoch in range(self.epochs):
            self.model.train()

            for batch in train_loader:
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                optimizer.step()
                scheduler.step()  # Update learning rate scheduler
                optimizer.zero_grad()
                progress_bar.update(1)

            self.model.eval()
            losses = []
            for step, batch in enumerate(test_loader):
                with torch.no_grad():
                    outputs = self.model(**batch)
                
                loss = outputs.loss
                losses.append(self.accelerator.gather(loss.repeat(self.batch_size)))
            
            losses = torch.cat(losses)
            losses = losses[:len(test_data)]

            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")
            print(f"Epoch: {epoch} | Loss: {torch.mean(losses)} | Perplexity: {perplexity}")

            # Save model
            if output_dir is not None:
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=self.accelerator.save)
                if self.accelerator.is_main_process:
                    self.tokenizer.save_pretrained(output_dir)






class RobertaEmbedding:
    """
    This class is used to infer vector embeddings from a sentence.

    Methods
    -------
        __init__(pretrained_model_path:Union[str, Path] = None)
            The constructor for the VectorEmbeddings class.
        _roberta_case_preparation()
            This method is used to prepare the Roberta model for the inference.
        infer_vector(doc:str, main_word:str)
            This method is used to infer the vector embeddings of a word from a sentence.
        infer_mask_logits(doc:str)
            This method is used to infer the logits of a word from a sentence.
    """
    def __init__(
        self,
        pretrained_model_path:Union[str, Path] = None,
    ):
        self.model_path = pretrained_model_path
        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise ValueError(
                    f'The path {pretrained_model_path} does not exist'
                )
            self.model_path = Path(pretrained_model_path)

        self._tokens = []
        self.model = None
        self.vocab = False

        lg.set_verbosity_error()
        self._roberta_case_preparation()

    @property
    def tokens(self):
        return self._tokens

    def _roberta_case_preparation(self) -> None:
        """
        This method is used to prepare the BERT model for the inference.
        """
        model_path = self.model_path if self.model_path is not None else 'roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = RobertaModel.from_pretrained(
            model_path, 
            output_hidden_states=True
            )
        self.MLM = RobertaForMaskedLM.from_pretrained(
            model_path,
            output_hidden_states = True,
        )
        self.max_length = self.model.config.max_position_embeddings
        self.model.eval()
        self.MLM.eval()
        self.vocab = True

    def infer_vector(self, doc:str, main_word:str) -> torch.Tensor:
        """
        This method is used to infer the vector embeddings of a word from a sentence.
        Args:
            doc: Document to process
            main_word: Main work to extract the vector embeddings for.

        Returns: 
            embeddings: Tensor of stacked embeddings (torch.Tensor) of shape (num_embeddings, embedding_size) where num_embeddings is the number of times the main_word appears in the doc.

        Examples:
            >>> model = RobertaEmbedding()
            >>> model.infer_vector(doc="The brown fox jumps over the lazy dog", main_word="fox")
            tensor([[-0.2182, -0.1597, -0.1723,  ..., -0.1706, -0.1709, -0.1709],
                    [-0.2182, -0.1597, -0.1723,  ..., -0.1706, -0.1709, -0.1709],
                    [-0.2182, -0.1597, -0.1723,  ..., -0.1706, -0.1709, -0.1709],
                    [-0.2182, -0.1597, -0.1723,  ..., -0.1706, -0.1709, -0.1709],
                    [-0.2182, -0.1597, -0.1723,  ..., -0.1706, -0.1709, -0.1709]])
        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        
     
        input_ids = self.tokenizer(doc, return_tensors="pt", max_length=self.max_length).input_ids
        token = self.tokenizer.encode(main_word, add_special_tokens=False)[0]

        word_token_index = torch.where(input_ids == token)[1]
        emb = []

        try:
            with torch.no_grad():
                embeddings = self.model(input_ids).last_hidden_state
               
            emb = [embeddings[0, idx] for idx in word_token_index]
            return torch.stack(emb)
        
        except:
            print(f'The word: "{main_word}" does not exist in the list of tokens')
            return torch.tensor(np.array(emb))



    
    def infer_mask_logits(self, doc:str) -> torch.Tensor:
        """
        This method is used to infer the logits of the mask token in a sentence.
        Args:
            doc (str): Document to process where the mask token is present.

        Returns: 
            logits: Tensor of stacked logits (torch.Tensor) of shape (num_embeddings, logits_size) where num_embeddings is the number of times the mask token (<mask>) appears in the doc.

        Examples:
            >>> model = RobertaEmbedding()
            >>> model.infer_mask_logits(doc="The brown fox <mask> over the lazy dog")
            tensor([[-2.1816e-01, -1.5967e-01, -1.7225e-01,  ..., -1.7064e-01,
                    -1.7090e-01, -1.7093e-01],
                    [-2.1816e-01, -1.5967e-01, -1.7225e-01,  ..., -1.7064e-01,
                    -1.7090e-01, -1.7093e-01],
                    [-2.1816e-01, -1.5967e-01, -1.7225e-01,  ..., -1.7064e-01,
                    -1.7090e-01, -1.7093e-01],
                    [-2.1816e-01, -1.5967e-01, -1.7225e-01,  ..., -1.7064e-01,
                    -1.7090e-01, -1.7093e-01],
                    [-2.1816e-01, -1.5967e-01, -1.7225e-01,  ..., -1.7064e-01,
                    -1.7090e-01, -1.7093e-01]])
        """

        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.MLM.__class__.__name__} has not been initialized'
            )

        input_ids = self.tokenizer(doc, return_tensors="pt", max_length=self.max_length).input_ids
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
        l = []
        try:
            with torch.no_grad():
                logits = self.MLM(input_ids).logits
                
            l = [logits[0, idx] for idx in mask_token_index]
            return torch.stack(l) if len(l) > 0 else torch.empty(0)
        
        except IndexError:
            raise ValueError(f'The mask falls outside of the max length of {self.max_length}, please use a smaller sentence')

        




class RobertaInference:
    """
    This class is used to infer vector embeddings from a sentence.

    Methods
    -------
        __init__(pretrained_model_path:Union[str, Path] = None)
            The constructor for the VectorEmbeddings class.
        _roberta_case_preparation()
            This method is used to prepare the Roberta model for the inference.
        get_embedding(word:str, sentence:str)
            This method is used to infer the vector embeddings of a word from a sentence.
        get_top_k_words(word:str, sentence:str, k:int=3)
            This method is used to infer the vector embeddings of a word from a sentence.
    """

    def __init__(
            self,
            pretrained_model_path:Union[str, Path] = None,
    ):
        self.model_path = pretrained_model_path
        if pretrained_model_path is not None:
            if not os.path.exists(pretrained_model_path):
                raise ValueError(
                    f'The path {pretrained_model_path} does not exist'
                )
            self.model_path = Path(pretrained_model_path)

        self.word_vectorizor = None
        self.vocab = False
        

        lg.set_verbosity_error()
        self._roberta_case_preparation()
    
    
    def _roberta_case_preparation(self) -> None:
        """
        This method is used to prepare the Roberta model for the inference.
        """
        model_path = self.model_path if self.model_path is not None else 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.word_vectorizor = RobertaEmbedding(pretrained_model_path=model_path)
        self.vocab = True

    
    def get_embedding(
            self,
            word : str, 
            sentence: Union[str, List[str]] = None,
            mask : bool = False
            ) -> torch.Tensor:
        
        """
        This method is used to infer the vector embeddings of a word from a sentence.
        Args:
            word: Word to get the vector embeddings for
            sentence: Sentence to get the vector embeddings from. If None, the word is assumed to be in the sentence. Defaults to None.
            mask: Whether to mask the word in the sentence or not. Defaults to False.
            
        Returns: 
            embeddings: Tensor of stacked embeddings (torch.Tensor) of shape (num_embeddings, embedding_size) where num_embeddings is the number of times the main_word appears in the doc, depending on the mask parameter.

        Examples:
            >>> model = RobertaInference()
            >>> model.get_embedding(word="office", sentence="The brown office is very big")
            Sentence:  The brown office is very big
            
            >>> model.get_embedding(word="office", sentence="The brown office is very big", mask=True)
            Sentence:  The brown <mask> is very big
        """
        
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        
        if sentence is None:
            sentence = ' ' + word.strip() + ' '
            
        if mask:
            sentence = sentence.replace(word, self.tokenizer.mask_token)
            word = self.tokenizer.mask_token
        
        else:
            word = ' ' + word.strip()
            
        embeddings = self.word_vectorizor.infer_vector(doc=sentence, main_word=word)
        return embeddings

    def get_top_k_words(
            self,
            word : str,
            sentence: str,
            k: int = 3
            ):
        """
        This method is used to infer the vector embeddings of a word from a sentence.
        Args:
            word: Word to mask
            sentence: Sentence to mask the word in
            k: Number of top words to return

        Returns:
            top_k_words: list

        Examples:
            >>> model = RobertaInference()
            >>> model.get_top_k_words(word="office", sentence="The brown office is very big")
            ['room', 'eye', 'bear']
        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        
        masked_sentence = sentence.replace(word, '<mask>')
        try:
            logits = self.word_vectorizor.infer_mask_logits(doc=masked_sentence)

            top_k = []

            for logit_set in logits:
                top_k_tokens = torch.topk(logit_set, k).indices
                top_k_words = [self.tokenizer.decode(token.item()).strip() for token in top_k_tokens]
                
                top_k.extend(top_k_words)

            return top_k
        
        except ValueError:
            return []




if __name__ == "__main__":
    # model = RobertaEmbedding(
    #     pretrained_model_path= "../../output/MLM_roberta_1980"
    # )
    
    # sentence = "The brown office is very big"
    # main_word = " office"

    # # emb = model.infer_mask_logits(doc=sentence)
    # # print(emb.shape)

    # model = RobertaInference(
    #     pretrained_model_path= "../../output/MLM_roberta_1980"
    # )

    # # top_k = model.get_top_k_words(
    # #     word="office",
    # #     sentence=sentence
    # # )
    # # print(top_k)

    # emb = model.get_embedding(
    #     word="office",
    #     sentence=sentence,
    #     mask=False
    # )
    # print(emb.shape)


    model = RobertaTrainer(
        model_name="roberta-base",
        max_length=128,
        mlm_probability=0.15,
        batch_size=4,
        learning_rate=1e-5,
        epochs=3,
        warmup_steps=500,
        split_ratio=0.8
    )

    model.train(
        data=["The brown fox jumps over the lazy dog", "The brown fox jumps over the lazy dog", "Hello world!"],
        output_dir="../../output/MLM_roberta_1980"
    )

