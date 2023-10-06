import os.path
import logging
import tqdm
from typing import Union, List
from pathlib import Path
import math
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup, logging as lg, pipeline
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from accelerate import Accelerator
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
            padding=True,
            ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_data = tokenizer(data, truncation=truncation, padding=padding, max_length=max_length)

    def __len__(self):
        return len(self.tokenized_data.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.tokenized_data.input_ids[idx], "labels": self.tokenized_data.input_ids[idx]}


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
    def __init__(self, model_name="roberta-base", max_length=128, mlm_probability=0.15, batch_size=4, learning_rate=1e-5, epochs=3, warmup_steps=500, split_ratio=0.8):
        # ... (same as before)
        self.split_ratio = split_ratio
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.accelerator = Accelerator()

    def prepare_dataset(self, data):
        dataset = CustomDataset(data, self.tokenizer, max_length=self.max_length)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
        return train_loader

    def train(
            self, 
            data,
            output_dir: Union[str, Path] = None
            ):
        train_data, test_data = train_test_split(data, test_size=1 - self.split_ratio, random_state=42)
        train_loader = self.prepare_dataset(train_data)
        test_loader = self.prepare_dataset(test_data)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        model, optimizer, train_loader, test_loader = self.accelerator.prepare(self.model, optimizer, train_loader, test_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=len(train_loader) * self.epochs)

        progress_bar = tqdm.tqdm(range(len(train_loader) * self.epochs), desc="Training", dynamic_ncols=True)
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






class VectorEmbeddings:
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
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForMaskedLM.from_pretrained(
            model_path,
            output_hidden_states = True,
        )
        self.model.eval()
        self.vocab = True

    def infer_vector(self, doc:str, main_word:str):
        """
        This method is used to infer the vector embeddings of a word from a sentence.
        Args:
            doc: Document to process
            main_word: Main work to extract the vector embeddings for.

        Returns: torch.Tensor

        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        
     
        tokenized_input = self.roberta_tokenizer(doc, return_tensors='pt')
        try:
            token_index = tokenized_input.index(self.roberta_tokenizer.encode(main_word)[0])
            with torch.no_grad():
                outputs = self.model(**tokenized_input)
                last_hidden_states = outputs.last_hidden_state
                main_token_embedding = last_hidden_states[:, token_index, :]

            return main_token_embedding

        except ValueError:
            raise ValueError(
                f'The word: "{main_word}" does not exist in the list of tokens: {tokenized_input} from {doc}'
            )




class MaskedWordInference:
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

        self.model = None
        self.vocab = False

        lg.set_verbosity_error()
        self._roberta_case_preparation()
    
    
    def _roberta_case_preparation(self) -> None:
        """
        This method is used to prepare the BERT model for the inference.
        """
        model_path = self.model_path if self.model_path is not None else 'roberta-base'
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForMaskedLM.from_pretrained(
            model_path,
            output_hidden_states = True,
        )
        self.model.eval()
        self.vocab = True

    
    def get_mask_embedding(
            self,
            word : str, 
            sentence: str
            ):
        
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        
        masked_sentence = sentence.replace(word, self.roberta_tokenizer.mask_token)
        tokenized_input = self.roberta_tokenizer(masked_sentence, return_tensors='pt')
        mask_token_index = torch.where(tokenized_input["input_ids"] == self.roberta_tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = self.model(**tokenized_input)
            last_hidden_states = outputs.last_hidden_state  # Get the last hidden states of the tokens
            mask_token_embedding = last_hidden_states[:, mask_token_index, :]
        return mask_token_embedding
    

    def get_top_k_words(
            self,
            word : str,
            sentence: str,
            k: int = 10
            ):
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self.model.__class__.__name__} has not been initialized'
            )
        

        masked_sentence = sentence.replace(word, self.roberta_tokenizer.mask_token)
        tokenized_input = self.roberta_tokenizer(masked_sentence, return_tensors='pt')
        mask_token_index = torch.where(tokenized_input["input_ids"] == self.roberta_tokenizer.mask_token_id)[1]

        with torch.no_grad():
            logits = self.model(**tokenized_input).logits
            mask_token_logits = logits[0, mask_token_index, :]
            top_k_tokens = torch.topk(mask_token_logits, k, dim=1).indices[0].tolist()

            top_k_words = []
            filled_sentences = []
            for token in top_k_tokens:
                w = self.roberta_tokenizer.decode([token])
                top_k_words.append(w)
                filled_sentences.append(masked_sentence.replace(self.roberta_tokenizer.mask_token, w))
            
            return top_k_words, filled_sentences
        


