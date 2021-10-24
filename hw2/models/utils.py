import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

def get_positions(self, tensor, padding_idx):
    ## TODO: check  + padding_idx
    mask = tensor.ne(padding_idx).int()
    position_tensor = (torch.cumsum(mask, dim = 1) * mask).long
    return position_tensor


