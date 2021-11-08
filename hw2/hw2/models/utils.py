import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

def get_positions(tensor, padding_idx):
    ## TODO: check  + padding_idx
    mask = tensor.ne(padding_idx).int()
    position_tensor = (torch.cumsum(mask, dim = 1) * mask).long()
    return position_tensor

def move_to_target_device(object, device):
    if torch.is_tensor(object):
        return object.to(device)
    elif isinstance(object, dict):
        return {k: move_to_target_device(v, device) for k, v in object.items()}
    elif isinstance(object, list):
        return [move_to_target_device(x, device) for x in object]
    else:
        return object

def strip_pad(tensor, pad, eos):
    # print(tensor)
    eos_idx = (tensor == eos).nonzero()
    if len(eos_idx) == 0:
        tensor = tensor
    else:
        tensor = tensor[:eos_idx[0][0]]
    return tensor[tensor.ne(pad)]




