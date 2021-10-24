import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from .bairu_encoder import BairuEncoder
from .bairu_config import BairuConfig

from .utils import get_positions

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BairuTransformerEncoder(BairuEncoder):
    def __init__(self, config: BairuConfig, dictionary, token_embedding = None):
        super().__init__(dictionary)
        num_tokens = dictionary.num_tokens
        self.embedding_dim = config.embedding_dim
        self.padding_idx = dictionary.pad()
        self.hidden_size = config.hidden_size
        if token_embedding is None:
            self.token_embedding = nn.Embedding(num_embeddings = num_tokens, embedding_dim = self.embedding_dim, padding_idx = self.padding_idx, )

        else:
            self.token_embedding = token_embedding

        if config.layernorm_embedding:
            self.layernorm_embedding = torch.nn.LayerNorm(self.embedding_dim)
        else:
            self.layernorm_embedding = None
    
        if config.position_embedding_type == 'absolute':
            self.positional_embedding = nn.Embedding(num_embeddings = config.max_position_embeddings, embedding_dim = self.embedding_dim)
        else:
            self.positional_embedding = None

        self.dropout = nn.Dropout(p = config.hidden_dropout_prob)
        self.batch_first = config.batch_first

        self.encoder_layer = TransformerEncoderLayer(d_model = self.hidden_size,nhead = config.num_attention_heads, dim_feedforward = config.intermediate_size, 
                                                        dropout = config.hidden_dropout_prob, activation = config.hidden_act, 
                                                        layer_norm_eps = config.layer_norm_eps, batch_first = self.batch_first)

        self.encoder = TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = config.num_hidden_layers)
    

    def forward_embedding(self, src_tokens,):
        x = embed = self.token_embedding(src_tokens)
        if self.positional_embedding is not None:
            positions = get_positions(src_tokens, self.padding_idx)
            x = embed + self.positional_embedding(positions)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout(x)
        return x, embed


    def forward(self, src_tokens: torch.LongTensor, src_length: torch.LongTensor, **kwargs):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        x, embed = self.forward_embedding(src_tokens)
        x = x * (1- encoder_padding_mask.unsqueeze(-1).type_as(x))
        if not self.batch_first:
            x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask = encoder_padding_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)

        return {
            'encoder_out': x,
            'encoder_padding_mask': encoder_padding_mask,
            'encoder_embedding': embed,
        }


