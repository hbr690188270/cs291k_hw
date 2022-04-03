import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from .bairu_encoder import BairuEncoder
from .bairu_config import BairuConfig
from ..module.embedding import TransformerEmbedding
from ..data.bairu_dictionary import dictionary

from .utils import get_positions

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BairuTransformerEncoder(BairuEncoder):
    def __init__(self, config: BairuConfig, src_dict: dictionary, ):
        '''
        mainly refers to fairseq implementation of Transformer encoder; but also rewrite the TransformerEmbedding module according to huggingface's BERT implementation
        '''
        super().__init__(src_dict)
        num_tokens = src_dict.num_tokens
        self.embedding_dim = config.embedding_dim
        self.padding_idx = src_dict.pad()
        self.hidden_size = config.hidden_size
        self.embedding = TransformerEmbedding(num_tokens, config.hidden_size, pad_token_id = src_dict.pad(), max_position_embeddings = config.max_position_embeddings, hidden_dropout_prob = config.hidden_dropout_prob,
                                             layer_norm_eps = config.layer_norm_eps, pe_type = config.decoder_pe, layernorm_embedding = config.layernorm_embedding)

        self.dropout = nn.Dropout(p = config.hidden_dropout_prob)
        self.batch_first = config.batch_first

        # self.encoder_layer = TransformerEncoderLayer(d_model = self.hidden_size,nhead = config.num_attention_heads, dim_feedforward = config.intermediate_size, 
        #                                                 dropout = config.hidden_dropout_prob, activation = config.hidden_act, 
        #                                                 layer_norm_eps = config.layer_norm_eps, batch_first = True)

        transformer_layer = TransformerEncoderLayer(d_model = self.hidden_size,nhead = config.num_attention_heads, dim_feedforward = config.intermediate_size,
                                    dropout = config.hidden_dropout_prob, activation = 'relu', batch_first = config.batch_first
                                    )

        self.encoder = TransformerEncoder(transformer_layer, num_layers = config.num_hidden_layers)        
    

    def forward_embedding(self, src_tokens,):
        x = embed = self.embedding(src_tokens)
        return x, embed


    def forward(self, src_tokens: torch.LongTensor, src_length: torch.LongTensor, **kwargs):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        x, embed = self.forward_embedding(src_tokens)
        x = x * (1- encoder_padding_mask.unsqueeze(-1).type_as(x))
        x = self.encoder(x, src_key_padding_mask = encoder_padding_mask)
        return {
            'encoder_out': x,
            'encoder_padding_mask': encoder_padding_mask,
            'encoder_embedding': embed,
        }


