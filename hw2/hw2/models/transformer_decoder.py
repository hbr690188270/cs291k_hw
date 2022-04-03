import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from .bairu_decoder import BairuDecoder
from .bairu_config import BairuConfig
from ..module.embedding import TransformerEmbedding
from ..data.bairu_dictionary import dictionary

from .utils import get_positions

from torch.nn import TransformerDecoder, TransformerDecoderLayer

class Decoder(nn.Module):
    def __init__(self, config: BairuConfig, tgt_dict: dictionary):
        super().__init__()
        num_tokens = tgt_dict.num_tokens
        self.embedding_dim = config.embedding_dim
        self.padding_idx = tgt_dict.pad()
        self.hidden_size = config.hidden_size
        self.nhead = config.num_attention_heads

        self.embedding = TransformerEmbedding(vocab_size = num_tokens, hidden_size = self.hidden_size, pad_token_id = config.pad_token_id, pe_type = config.decoder_pe)

        transformer_layer = TransformerDecoderLayer(d_model = self.hidden_size,nhead = config.num_attention_heads, dim_feedforward = config.intermediate_size,
                                    dropout = config.hidden_dropout_prob, activation = 'relu',  batch_first = config.batch_first
                                    )
        self.decoder = TransformerDecoder(transformer_layer, num_layers = config.num_hidden_layers)
        self.output_layer = torch.nn.Linear(self.hidden_size, num_tokens)
    
    def build_decoder_attention_mask(self, prev_output_tokens, window_size = 25, seq_lengths = None):
        batch_size, seq_len = prev_output_tokens.size()
        attention_mat_list = []
        if seq_len <= window_size or window_size <= 0:
            attention_mat = torch.triu(torch.ones([seq_len, seq_len], device = prev_output_tokens.device, dtype = torch.long), diagonal = 1).bool()
        else:
            for batch_idx in range(batch_size):
                attention_mat = torch.triu(torch.ones([seq_len, seq_len], device = prev_output_tokens.device, dtype = torch.long), diagonal = 1)
                if seq_lengths[batch_idx] <= window_size:
                    pass
                else:
                    to_mask_size = seq_lengths[batch_idx] - window_size
                    for i in range(window_size, seq_len):
                        attention_mat[i][:to_mask_size] = 1
                attention_mat_list.append(attention_mat)
            attention_mat = torch.stack(attention_mat_list, dim = 0)
            attention_mat = attention_mat.bool()
            attention_mat = attention_mat.unsqueeze(1).repeat(1, self.nhead, 1,1).reshape(-1, seq_len, seq_len)
        
        return attention_mat


    def forward(self, prev_output_tokens:torch.LongTensor, encoder_out = None, **kwargs):
        encoder_output = encoder_out['encoder_out']
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        x, embed = self.embedding(prev_output_tokens)
        tgt_padding_mask = prev_output_tokens.eq(self.padding_idx)
        seq_lengths = torch.sum(1-tgt_padding_mask.long(), dim = 1)
        # if self.decoder_att_window_size > 0:
        tgt_mask = self.build_decoder_attention_mask(prev_output_tokens, window_size = -1, seq_lengths = seq_lengths)
        # else:
        #     tgt_mask = None
        output = self.decoder(tgt = x, memory = encoder_output, tgt_mask = tgt_mask, memory_key_padding_mask = encoder_padding_mask, tgt_key_padding_mask = tgt_padding_mask)
        output = self.output_layer(output)
        return {
            'output':output
            }

