import os.path
import torch
from pathlib import Path
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import logging as lg
import logging
from typing import Optional, Union


class BertEmbeddings:
    """
    This class is used to infer the vector embeddings of a word from a sentence.

    Methods
    -------
        infer_vector(doc:str, main_word:str)
            This method is used to infer the vector embeddings of a word from a sentence.
        _bert_case_preparation()
            This method is used to prepare the BERT model for the inference.
    """
    def __init__(
        self,
        pretrained_model_path: Optional[Union[str, Path]] = None,
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
        self.lematizer = None

        lg.set_verbosity_error()
        self._bert_case_preparation()

    @property
    def tokens(self):
        return self._tokens

    def _bert_case_preparation(self) -> None:
        """
        This method is used to prepare the BERT model for the inference.
        """
        model_path = self.model_path if self.model_path is not None else 'bert-base-uncased'
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(
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
        marked_text = "[CLS] " + doc + " [SEP]"
        tokens = self.bert_tokenizer.tokenize(marked_text)
        try:
            main_token_id = tokens.index(main_word.lower())
            idx = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            segment_id = [1] * len(tokens)

            self.tokens_tensor = torch.tensor([idx])
            self.segments_tensors = torch.tensor([segment_id])

            with torch.no_grad():
                outputs = self.model(self.tokens_tensor, self.segments_tensors)
                hidden_states = outputs[2]

            return hidden_states[-2][0][main_token_id]

        except ValueError:
            raise ValueError(
                f'The word: "{main_word}" does not exist in the list of tokens: {tokens} from {doc}'
            )


class BertInference:
    def __init__(self) -> None:
        pass

    def get_top_k_words(self):
        pass



if __name__ == '__main__':
    pass
