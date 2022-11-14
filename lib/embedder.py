import typing

import numpy as np
from transformers import BertTokenizer, BertModel


class Embedder():
    """ Generate embeddings using the BERT multi-lingual encoder """
    def __init__(self, max_len=30):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.max_len = max_len

    def transform(self, texts: typing.List) -> np.array:
        """
        Generate embeddings for the input list of texts

        Args:
             texts: list of sentences of length <batch>

        Returns:
            numpy array of shape <batch, tokens, embed_dim>
        """
        enc_input = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True,
                                   max_length=self.max_len)
        output = self.model(**enc_input)
        return output.last_hidden_state.detach().numpy()  # B x L x D
