import math
import argparse
from os import read

class BLEUMetric():
    def __init__(self, n_gram_list = [1,2,3,4], pow_list = [1/6, 1/3, 1/6, 1/3]):
        self.n_gram_list = n_gram_list
        self.pow_list = pow_list

    def cal_sentence_pn(self, hyp, ref, n_gram = 1):
        '''
        hyp/ref: list of str
        '''
        n_gram_hyp = self.cal_sentence_ngram(hyp, n_gram)
        n_gram_ref = self.cal_sentence_ngram(ref, n_gram)
        total_num = sum([x for x in n_gram_hyp.values()])
        correct_num = 0
        for gram, freq in n_gram_hyp.items():
            if gram in n_gram_ref:
                # correct_num += freq
                correct_num += min([freq, n_gram_ref[gram]])
        pn = correct_num/total_num
        # print("{} gram: ".format(n_gram), pn)
        return pn, correct_num, total_num

    def cal_sentence_ngram(self, sentence, n_gram = 1):
        '''
        corpus: list of str
        '''
        n_gram_dict = {}
        word_list = sentence.split()
        for i in range(len(word_list) - n_gram + 1):
            gram_set = tuple([word_list[i + j] for j in range(n_gram)])
            if gram_set in n_gram_dict:
                n_gram_dict[gram_set] += 1
            else:
                n_gram_dict[gram_set] = 1
        return n_gram_dict

    def brevity(self, hyp_len, ref_len):
        r = ref_len / hyp_len
        return min([1, math.exp(2 - 2 * r)])
    
    def cal_len(self, corpus):
        return sum([len(x.split()) for x in corpus])

    def __call__(self, hyp, ref):
        hyp_len = self.cal_len(hyp)
        ref_len = self.cal_len(ref)
        bleu_score = self.brevity(hyp_len, ref_len)
        print("penalty: ", bleu_score)
        for i in range(len(self.n_gram_list)):
            n_gram = self.n_gram_list[i]
            pow_value = self.pow_list[i]
            sum_correct = sum_total = 0
            for j in range(len(hyp)):
                pn, correct_num, total_num = self.cal_sentence_pn(hyp[j], ref[j], n_gram)
                sum_correct += correct_num
                sum_total += total_num
            pn = sum_correct / sum_total
            bleu_score *= math.pow(pn, pow_value)
        return bleu_score
