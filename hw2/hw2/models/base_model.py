import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from .bairu_encoder import BairuEncoder
from .bairu_decoder import BairuDecoder

class BairuEncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert isinstance(self.encoder, BairuEncoder)
        assert isinstance(self.decoder, BairuDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens = None, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out = encoder_out, **kwargs) 
        return decoder_out
    
    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

