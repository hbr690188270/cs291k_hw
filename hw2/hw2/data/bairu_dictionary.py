import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

class dictionary():
    def __init__(self, bos = "<s>", pad = "<pad>", eos = "</s>", unk = "<unk>",):
        self.word2idx = {}
        self.idx2word = {}
        self.word2freq = {}
        self.word_list = []
        self.bos_word = bos
        self.eos_word = eos
        self.unk_word = unk
        self.pad_word = pad

        self.pad_index = self.add_word(self.pad_word)
        self.bos_index = self.add_word(self.bos_word)
        self.eos_index = self.add_word(self.eos_word)
        self.unk_index = self.add_word(self.unk_word)


    def add_word(self, word):
        if word in self.word2idx:
            idx = self.word2idx[word]
            self.word2freq[word] += 1
            return idx
        else:
            idx = len(self.word_list)
            self.word2idx[word] = idx
            self.word2freq[word] = 1
            self.word_list.append(word)
            self.idx2word[idx] = word
            return idx

    def bos(self):
        return self.bos_index

    def eos(self):
        return self.eos_index

    def pad(self):
        return self.pad_index

    def unk(self):
        return self.unk_index

    def get_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]    
        else:
            return self.unk()

    def build_vocab(self, filelist):
        for filename in filelist:
            with open(filename, 'r', encoding = 'utf-8') as f:
                for line in f:
                    word_list = line.strip().split()
                    for word in word_list:
                        self.add_word(word)
    @property
    def num_tokens(self):
        return len(self.word_list)

    def read_from_vocabfile(self, file_addr):
        # with open(file_addr, 'r', encoding = 'utf-8') as f:
        raise NotImplementedError

    def save_pretrained(self, save_dir = './dictionary.pkl'):
        import pickle
        with open(save_dir, 'wb') as f:
            pickle.dump(self, f)

    def decode(self, idx_list):
        '''
        idx_list: list
        '''
        tokens = [self.idx2word[x] for x in idx_list]
        if len(tokens) == 0:
            return ""
        strings = []
        word = ""
        for i in range(len(tokens)):
            if tokens[i][-2:] == '@@': 
                word += tokens[i][:-2]
            else:
                word += tokens[i]
                strings.append(word)
                word = ""
            # else:

        return ' '.join(strings)

