import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

class BairuDecoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
    
    def forward(self, prev_output_tokens, encoder_out = None, **kwargs):
        x, extra = self.extract_features(prev_output_tokens, encoder_out, **kwargs)
        res = self.output_layer(x,)
        return res, extra

    
    def extract_features(self, prev_output_tokens, encoder_out = None, **kwargs):
        raise NotImplementedError
    
    def output_layer(self, features, **kwargs):
        raise NotImplementedError


