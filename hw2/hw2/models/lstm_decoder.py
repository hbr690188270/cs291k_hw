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
        self.decoder_transform = nn.Linear(config.decoder_hidden_size, config.hidden_size)
        
        if config.hidden_size != config.decoder_hidden_size:
            self.encoder_transform = nn.Linear(config.hidden_size, config.decoder_hidden_size,)
        else:
            self.encoder_transform = None

    def forward(self, decoder_hidden_state, encoder_out, encoder_padding_mask = None):
        '''
        decoder_hidden_state: [batch_size, decoder_hidden_dim]   ONLY one time step!
        encoder_out: [enc_seq_len, batch_size , encoder_output_dim]   batch_first = False!
        encoder_padding_mask: [enc_seq_len, batch_size, encoder_output_dim]  batch_first = False!
        '''
        x = self.decoder_transform(decoder_hidden_state)
        weight = (x.unsqueeze(0) * encoder_out).sum(dim = 2)
        if encoder_padding_mask != None:
            weight = weight.float().masked_fill_(encoder_padding_mask, -10000)
        
        attention_score = torch.unsqueeze(F.softmax(weight, dim = 0), dim = 2)  ## seq_len  batch_size  1
        x = (attention_score * encoder_out).sum(0)
        if self.encoder_transform is not None:
            x = self.output_proj(x) + decoder_hidden_state  ## something like residual connection
        else:
            x = x + decoder_hidden_state
        return x


class BairuLSTMDecoder(BairuDecoder):
    def __init__(self, config: BairuConfig, dictionary, token_embedding = None):
        super().__init__(dictionary)
        num_tokens = dictionary.num_tokens
        self.padding_idx = config.pad_token_id
        self.hidden_size = config.decoder_hidden_size
        self.hidden_layer = config.decoder_hidden_layer
        self.embedding_dim = config.decoder_embedding_dim
        self.num_layers = config.decoder_hidden_layer
        self.batch_first = config.batch_first
        self.input_feed_size = config.decoder_hidden_size
        self.decoder_init = config.decoder_init
        if token_embedding is None:
            self.token_embedding = nn.Embedding(num_embeddings = num_tokens, embedding_dim = self.embedding_dim, padding_idx = self.padding_idx, )
        else:
            self.token_embedding = token_embedding
        self.LSTMLayers = nn.ModuleList(
            [
                LSTMCell(input_size = config.embedding_dim + self.input_feed_size, hidden_size = config.decoder_hidden_size)
                if layer == 0 else LSTMCell(input_size = config.decoder_hidden_size , hidden_size = config.decoder_hidden_size)
                for layer in range(config.decoder_hidden_layer)
                ]
        )
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
        net_output = {
            'output':res,
            'extra':extra
        }
        return net_output
    
    def extract_features(self, prev_output_tokens, encoder_out = None, **kwargs):
        encoder_output = encoder_out['encoder_out']
        encoder_padding_mask = encoder_out['encoder_padding_mask']

        batch_size, seq_len = prev_output_tokens.size()
        x, embed = self.forward_embedding(prev_output_tokens)
        if not self.batch_first:
            x = x.transpose(0, 1)
            encoder_output = encoder_output.transpose(0,1)
            encoder_padding_mask = encoder_padding_mask.transpose(0,1)
        if self.decoder_init == 'enc':
            if self.batch_first:
                enc_init = torch.mean(encoder_output, dim = 1)
            else:
                enc_init = torch.mean(encoder_output, dim = 0)
                h0 = [enc_init for _ in range(self.hidden_layer)]
                c0 = [enc_init for _ in range(self.hidden_layer)]            
        elif self.decoder_init == 'none':
            zero_state = x.new_zeros(batch_size, self.hidden_size)
            h0 = [zero_state for _ in range(self.hidden_layer)]
            c0 = [zero_state for _ in range(self.hidden_layer)]
        else:
            raise Exception()
        ## Please refer to paper  Effective Approaches to Attention-based Neural Machine Translation  for the input feed operation 
        prev_hidden = x.new_zeros(batch_size, self.hidden_size)


        outs = []
        for j in range(seq_len):
            input_x = torch.cat([x[j,:,:], prev_hidden], dim = 1)
            for i, lstm_cell in enumerate(self.LSTMLayers):
                h, c = lstm_cell(input_x,(h0[i], c0[i]))

                input_x = self.dropout(h)
                if self.residual:
                    input_x = input_x + h0[i]
                h0[i] = h
                c0[i] = c

            ## Attention
            out = self.dropout(self.LSTMAttention(h, encoder_output, encoder_padding_mask))
            outs.append(out)
            prev_hidden = out
        x = torch.stack(outs, dim = 0)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x, []
        # return {
        #     'output':x,
        #     'others':[]
        # }

    def output_layer(self, features, **kwargs):
        x = self.output_projection(features)
        return x
 

