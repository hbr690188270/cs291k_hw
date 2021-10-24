import torch
import torch.nn.functional as F
import numpy as np
from ..data.bairu_dictionary import dictionary
from ..models.base_model import BairuEncoderDecoderModel

class CrossEntropyLossMetric():
    def __init__(self, data_dict:dictionary, report_accuracy = True):
        self.data_dict = data_dict
        self.report_accuracy = report_accuracy
    
    def __call__(self, model: BairuEncoderDecoderModel, batch_data, reduction = 'mean'):
        net_input = batch_data['net_input']
        net_output = model(**net_input)
        logits = net_output['output']
        target = batch_data['target']
        loss = F.cross_entropy(logits, target, ignore_index = self.data_dict.pad(), reduction = reduction)
        
        if self.debug:
            print("pred: ", torch.argmax(logits, dim = -1))
            print("target: ", target)
            print()
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            logits.argmax(-1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)    
        accuracy = n_correct / total
        return loss, n_correct, total


