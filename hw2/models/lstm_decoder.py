import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from .bairu_decoder import BairuDecoder
from .bairu_config import BairuConfig
from torch.nn import LSTMCell, LSTM
from transformers import BertModel

class LSTMCrossAttention(nn.Module):
    def __init__(self, config:BairuConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.decoder_hidden_size, config.hidden_size)
        self.output_proj = nn.Linear(
            config.decoder_hidden_size + config.hidden_size, config.decoder_hidden_size,
        )

    def forward(self, decoder_hidden_state, encoder_out, encoder_padding_mask = None):
        '''
        decoder_hidden_state: [batch_size, decoder_hidden_dim]   ONLY one time step!
        encoder_out: [enc_seq_len, batch_size , encoder_output_dim]   batch_first = False!
        encoder_padding_mask: [enc_seq_len, batch_size, encoder_output_dim]  batch_first = False!
        '''
        x = self.input_proj(decoder_hidden_state)
        weight = (x.unsqueeze(0) * decoder_hidden_state).sum(dim = 2)
        if encoder_padding_mask != None:
            weight = weight.float().masked_fill_(encoder_padding_mask, float("-inf"))
        
        attention_score = F.softmax(weight, dim = 0)  ## seq_len  batch_size
        x = (attention_score.unsqueeze(2) * encoder_out).sum(0)
        x = self.output_proj(x)
        return x, attention_score


class BairuLSTMDecoder(BairuDecoder):
    def __init__(self, config: BairuConfig, dictionary,):
        super().__init__(dictionary)
        num_tokens = dictionary.num_tokens
        self.padding_idx = config.pad_token_id
        self.hidden_size = config.decoder_hidden_size
        self.hidden_layer = config.decoder_hidden_layer
        self.token_embedding = nn.Embedding(num_embeddings = num_tokens, embedding_dim = self.embedding_dim, padding_idx = self.padding_idx, )

        self.LSTMLayers = nn.ModuleList(
            [
                LSTMCell(input_size = config.embedding_dim, hidden_size = config.decoder_hidden_size)
                for layer in config.decoder_hidden_layer
                ]
        )
        self.LSTM = nn.LSTM(input_size = config.embedding_dim, hidden_size = config.decoder_hidden_size, num_layers = config.decoder_hidden_layer, 
                            batch_first = config.batch_first, dropout = config.hidden_dropout_prob, bidirectional = False)
        self.dropout = nn.Dropout(p = config.hidden_dropout_prob)
        self.LSTMAttention = LSTMCrossAttention(config,)
        self.output_projection = nn.Linear(config.decoder_hidden_size, num_tokens)
        

        self.residual = config.decoder_residual
        if config.layernorm_embedding:
            self.layernorm_embedding = torch.nn.LayerNorm(self.embedding_dim)
        else:
            self.layernorm_embedding = None
    

    def forward_embedding(self, prev_output_tokens,):
        x = embed = self.token_embedding(prev_output_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout(x)
        return x, embed

    def forward(self, prev_output_tokens, encoder_out = None, **kwargs):
        x, extra = self.extract_features(prev_output_tokens, encoder_out, **kwargs)
        res = self.output_layer(x,)
        return res, extra

    
    def extract_features(self, prev_output_tokens, encoder_out = None, **kwargs):
        encoder_output = encoder_out['encoder_out']
        encoder_padding_mask = encoder_out['encoder_padding_mask']

        batch_size, seq_len = prev_output_tokens.size()
        x, embed = self.forward_embedding(prev_output_tokens)
        x = x * (1- encoder_padding_mask.unsqueeze(-1).type_as(x))
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        zero_state = x.new_zeros(batch_size, self.hidden_size)
        prev_hiddens = [zero_state for _ in range(self.hidden_layer)]
        prev_cells = [zero_state for _ in range(self.hidden_layer)]
        ## Please refer to paper  Effective Approaches to Attention-based Neural Machine Translation  for the input feed operation 
        input_feed = x.new_zeros(batch_size, self.hidden_size)


        outs = []
        for j in range(seq_len):
            input_x = torch.cat([x[j,:,:], input_feed], dim = 1)
            for i, lstm_cell in enumerate(self.layers):
                hidden_state, cell_state = lstm_cell(input_x,(prev_hiddens[i], prev_cells[i]))
                input_x = self.dropout(hidden_state)
                if self.residual:
                    input_x = input_x + prev_hiddens[i]
                prev_hiddens[i] = hidden_state
                prev_cells = cell_state

            ## Attention
            out, _ = self.LSTMAttention(hidden_state, encoder_output, encoder_padding_mask)
            out = self.dropout(out)
            outs.append(out)
            input_feed = out
        x = torch.stack(outs, dim = 0)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x,[]

    def output_layer(self, features, **kwargs):
        x = self.output_projection(features)
        return x
 

