import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

class BairuEncoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, src_tokens, src_length = None, **kwargs):
        '''
        src_tokens: torch.LongTensor  tokens in the source language with shape [batch_size, src_len]
        src_length: torch.LongTensor   length of each source sentences with shape [batch_size, ]
        '''
        raise NotImplementedError
    
    def reorder_encoder_out(self, encoder_output, new_order):
        '''
        encoder_out: output from the forward function
        new_order: torch.LongTensor:  desired order
        '''
        raise NotImplementedError

    def max_positions(self):
        return 512
    
    


