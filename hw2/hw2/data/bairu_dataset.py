import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch

from .bairu_dictionary import dictionary

class MTDataset(Dataset):
    def __init__(self, src_dict:dictionary, tgt_dict:dictionary,src = None, tgt = None, src_corpus_dir = None, tgt_corpus_dir = None, 
                        left_pad = False, shuffle = True, append_eos_to_target = True, padding_strategy = "LONGEST", max_len = 100,
                        sanity_check = False):
        assert (src != None or src_corpus_dir != None)
        assert (tgt != None or tgt_corpus_dir != None)
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.sanity_check = sanity_check
        if src == None:
            self.src, src_error_index = self.read_corpus(src_corpus_dir)
        else:
            self.src = src
        if tgt == None:
            self.tgt, tgt_error_index = self.read_corpus(tgt_corpus_dir)
        else:
            self.tgt = tgt
        
        all_error_index = src_error_index.union(tgt_error_index)
        filtered_src, filtered_tgt = [],[]
        for i in range(len(self.src)):
            if i not in all_error_index:
                filtered_src.append(self.src[i])
                filtered_tgt.append(self.tgt[i])
        self.src = filtered_src
        self.tgt = filtered_tgt

        self.num_src = len(self.src)
        self.num_tgt = len(self.tgt)
        assert self.num_src == self.num_tgt
        self.left_pad = False
        self.shuffle = shuffle
        self.append_eos_to_target = append_eos_to_target

        self.padding_strategy = padding_strategy
        self.max_len = max_len


    def read_corpus(self, data_dir, src = True):
        ## TODO: split?
        data_list = []
        count = 0
        error_index = []
        with open(data_dir, 'r', encoding = 'utf-8') as f:
            for line in f:
                word_list = line.strip().split()
                if src:
                    idx_list = [self.src_dict.get_index(x) for x in word_list]
                else:
                    idx_list = [self.tgt_dict.get_index(x) for x in word_list]
                if len(word_list) == 0:
                    error_index.append(count)
                data_list.append(torch.LongTensor(idx_list))
                count += 1
                if self.sanity_check and count >= 100:
                    break
        return data_list, set(error_index)

    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = self.tgt[index]
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos()
            if tgt_item[-1] != eos:
                tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])

        data_instance = {
            'source': src_item,
            'target': tgt_item,
            'index': index,
        }
        return data_instance

    def __len__(self):
        return len(self.src)

    def collate(self, idx_tensors, move_eos_to_beginning = False):
        max_length = min([max([x.size(0) for x in idx_tensors]), self.max_len])
        # max_length = max([x.size(0) for x in idx_tensors])

        batch_size = len(idx_tensors)
        new_idx_tensors = idx_tensors[0].new(batch_size, max_length).fill_(self.src_dict.pad())
        for i in range(batch_size):
            orig_length = len(idx_tensors[i])
            if move_eos_to_beginning:
                new_idx_tensors[i][0] = self.tgt_dict.eos()
                if orig_length > max_length:
                    new_idx_tensors[i][1:max_length] = idx_tensors[i][:max_length - 1]
                else:
                    new_idx_tensors[i][1:orig_length] = idx_tensors[i][:-1]
            else:
                if orig_length > max_length:
                    new_idx_tensors[i][:max_length] = idx_tensors[i][:max_length]
                else:
                    new_idx_tensors[i][:orig_length] = idx_tensors[i]
        return new_idx_tensors

    def collater(self, data_list, ):
        index_list = [x['index'] for x in data_list]
        src_lengths = torch.LongTensor([x['source'].ne(self.src_dict.pad()).sum() for x in data_list])
        src_tokens = self.collate([x['source'] for x in data_list])
        tgt_tokens = self.collate([x['target'] for x in data_list])
        prev_output_tokens = self.collate([x['target'] for x in data_list], move_eos_to_beginning = True)

        batch_data = {
            'index': index_list,
            'num_samples': len(index_list),
            'net_input':{
                'src_tokens':src_tokens,
                'src_lengths': src_lengths,
                'tgt_tokens':tgt_tokens,
                'prev_output_tokens': prev_output_tokens,
            }
        }
        return batch_data

