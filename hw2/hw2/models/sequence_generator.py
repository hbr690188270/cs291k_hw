import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BairuEncoderDecoderModel
from .bairu_encoder import BairuEncoder
from .bairu_decoder import BairuDecoder
from ..data.bairu_dictionary import dictionary

class SequenceGenerator(nn.Module):
    def __init__(self, model: BairuEncoderDecoderModel, tgt_dict: dictionary, beam_size: int, max_length: int, beginning = 'eos'):
        super().__init__()
        self.model = model
        self.tgt_dict = tgt_dict
        self.beam_size = beam_size
        self.max_length = max_length
        if beginning == 'eos':   ## keep the same begining token with training phase
            self.begin_token = self.tgt_dict.eos()
        else:
            self.begin_token = self.tgt_dict.bos()

    def log_probs(self, logits):
        return F.log_softmax(logits, dim = -1)[:, -1, :]

    def beam_search(self, predict_prob, scores, generated_tokens, position):
        '''
        predict_prob: batch_size * beam_size, vocab_num
        scores: batch_size, beam_size, max_length 

        add predict prob with prob score   first reshape the predict_prob into batch_size, beam_size, 1   then broadcast
        '''
        total_item, vocab_num = predict_prob.size()
        batch_size = total_item // self.beam_size
        # predict_prob = predict_prob.view(batch_size, 1, vocab_num).repeat(1, self.beam_size, 1)
        predict_prob = predict_prob.view(batch_size, self.beam_size, vocab_num)
        if position == 0:
            predict_prob = predict_prob
        else:
            predict_prob = predict_prob + scores[:,:, position - 1].unsqueeze(-1)    ## batch_size, beam_size, vocab_num

        top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size, dim = -1)
        top_scores, top_indices = top_pred
        beam_idx = top_indices // vocab_num
        word_idx = torch.fmod(top_indices, vocab_num)
        return top_scores, word_idx, beam_idx     

    def expand_encoder_output(self, encoder_output, new_order):
        encoder_output['encoder_out'] = encoder_output['encoder_out'].index_select(0, new_order)
        encoder_output['encoder_padding_mask'] = encoder_output['encoder_padding_mask'].index_select(0, new_order)
        return encoder_output

    def forward(self, batch_data,):
        net_input = batch_data['net_input']
        src_tokens = net_input['src_tokens']
        src_lengths = net_input['src_lengths']
        tgt_tokens = net_input['tgt_tokens']
        batch_size, seq_len = src_tokens.size()

        curr_device = src_tokens.device

        ## encoder forward
        encoder_output = self.model.encoder(src_tokens, src_lengths)
        encoder_out = encoder_output['encoder_out'] 
        encoder_padding_mask = encoder_output['encoder_padding_mask']
        encoder_hidden_dim = encoder_out.size(-1)

        ## expand the encoder_out to shape [batch_size * beam_size, seq_len, hidden_dim]
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        new_encoder_out = self.expand_encoder_output(encoder_output, new_order)




        ## start beam search
        generated_tokens = torch.zeros(batch_size * self.beam_size, self.max_length + 2).to(curr_device).long()
        scores = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device)

        bbsz_offsets = (
                    (torch.arange(0, batch_size) * self.beam_size)
                    .unsqueeze(1)
                    .type_as(generated_tokens)
                    .to(src_tokens.device)
                )


        final_scores = torch.zeros(batch_size * self.beam_size).to(curr_device)

        generated_tokens[:,0] = self.begin_token
        finished_flag = torch.zeros(batch_size * self.beam_size).to(curr_device).eq(-1)
        finished = [False for i in range(batch_size)]
        unfinished_sents = len(finished)

        for position in range(self.max_length):
            decoder_output = self.model.forward_decoder(generated_tokens[:, :position + 1], new_encoder_out)
            predict_probs = decoder_output['output']
            predict_probs = self.log_probs(predict_probs)
            predict_probs[:, self.tgt_dict.pad()] = -math.inf
            predict_probs[:, self.tgt_dict.unk()] = -math.inf
            predict_probs[:, self.tgt_dict.bos()] = -math.inf
            
            if position >= self.max_length - 1:
                predict_probs[:] = -math.inf
                predict_probs[:, self.tgt_dict.eos()] = 0
                generated_tokens[:, position + 1] = self.tgt_dict.eos()
                break
            if position >= 1:
                eos_mask = generated_tokens[:, position].eq(self.tgt_dict.eos())
                # print("tokens at position ", position, ": ", generated_tokens[:, :position])
                # print(eos_mask)
                predict_probs[eos_mask] = -math.inf
                predict_probs[eos_mask, self.tgt_dict.eos()] = 0
            # predict_probs[finished_flag, self.tgt_dict.pad()] = math.inf
            # print(predict_probs[finished_flag, self.tgt_dict.pad()])
            candidate_scores, candidate_idxs, candidate_beam_idxs = self.beam_search(predict_probs, scores.view(batch_size, self.beam_size, -1), generated_tokens, position)


            cand_bbsz_idx = candidate_beam_idxs + bbsz_offsets    ## batch， beam, 对应到generated token中的位置
            cand_bbsz_idx = cand_bbsz_idx.view(-1)


            generated_tokens = generated_tokens[cand_bbsz_idx]
            generated_tokens[:, position + 1] = candidate_idxs.view(-1)

            # generated_tokens[cand_bbsz_idx, position + 1] = candidate_idxs

            # print("beam search words: ", candidate_idxs)
            # print("beam search beam idx:", cand_bbsz_idx)
            # print("updated sequence: ")
            # print(generated_tokens[:, :position + 2])
            # pause = input("??")

            scores[:, position + 1] = candidate_scores.view(-1)
            final_scores = scores[:, position + 1]
            # final_scores[new_eos_mask] = scores[new_eos_mask, position]  ## use the score before <eos>
            
            finished_flag = generated_tokens[:, position + 1].eq(self.tgt_dict.eos())
            
            if finished_flag.sum() == batch_size * self.beam_size:
                break
            if position >= self.max_length:
                break
        
        batch_scores = final_scores.view(batch_size, self.beam_size)
        best_beam = torch.argmax(batch_scores, dim = -1)
        generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
        best_sentences = torch.stack([generated_tokens[i, best_beam[i]] for i in range(len(best_beam))], dim = 0)
        return best_sentences[:, 1:]


class SequenceGenerator_backup(nn.Module):
    def __init__(self, model: BairuEncoderDecoderModel, tgt_dict: dictionary, beam_size: int, max_length: int, beginning = 'eos'):
        super().__init__()
        self.model = model
        self.tgt_dict = tgt_dict
        self.beam_size = beam_size
        self.max_length = max_length
        if beginning == 'eos':   ## keep the same begining token with training phase
            self.begin_token = self.tgt_dict.eos()
        else:
            self.begin_token = self.tgt_dict.bos()

    def log_probs(self, logits):
        return F.log_softmax(logits, dim = -1)[:, -1, :]

    def beam_search(self, predict_prob, scores, generated_tokens, position):
        '''
        predict_prob: batch_size * beam_size, vocab_num
        scores: batch_size, beam_size, max_length 

        add predict prob with prob score   first reshape the predict_prob into batch_size, beam_size, 1   then broadcast
        '''
        total_item, vocab_num = predict_prob.size()
        batch_size = total_item // self.beam_size
        # predict_prob = predict_prob.view(batch_size, 1, vocab_num).repeat(1, self.beam_size, 1)
        predict_prob = predict_prob.view(batch_size, self.beam_size, vocab_num)
        if position == 0:
            predict_prob = predict_prob
        else:
            predict_prob = predict_prob + scores[:,:, position - 1].unsqueeze(-1)    ## batch_size, beam_size, vocab_num

        top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size, dim = -1)
        top_scores, top_indices = top_pred
        beam_idx = top_indices // vocab_num
        word_idx = torch.fmod(top_indices, vocab_num)
        return top_scores, word_idx, beam_idx     

    def expand_encoder_output(self, encoder_output, new_order):
        encoder_output['encoder_out'] = encoder_output['encoder_out'].index_select(0, new_order)
        encoder_output['encoder_padding_mask'] = encoder_output['encoder_padding_mask'].index_select(0, new_order)
        return encoder_output

    def forward(self, batch_data,):
        net_input = batch_data['net_input']
        src_tokens = net_input['src_tokens']
        src_lengths = net_input['src_lengths']
        tgt_tokens = net_input['tgt_tokens']
        batch_size, seq_len = src_tokens.size()

        curr_device = src_tokens.device

        ## encoder forward
        encoder_output = self.model.encoder(src_tokens, src_lengths)
        encoder_out = encoder_output['encoder_out'] 
        encoder_padding_mask = encoder_output['encoder_padding_mask']
        encoder_hidden_dim = encoder_out.size(-1)

        ## expand the encoder_out to shape [batch_size * beam_size, seq_len, hidden_dim]
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        new_encoder_out = self.expand_encoder_output(encoder_output, new_order)




        ## start beam search
        generated_tokens = torch.zeros(batch_size * self.beam_size, self.max_length + 2).to(curr_device).long()
        scores = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device)

        bbsz_offsets = (
                    (torch.arange(0, batch_size) * self.beam_size)
                    .unsqueeze(1)
                    .type_as(generated_tokens)
                    .to(src_tokens.device)
                )


        final_scores = torch.zeros(batch_size * self.beam_size).to(curr_device)

        generated_tokens[:,0] = self.begin_token
        finished_flag = torch.zeros(batch_size * self.beam_size).to(curr_device).eq(-1)
        finished = [False for i in range(batch_size)]
        unfinished_sents = len(finished)

        for position in range(self.max_length):
            decoder_output = self.model.forward_decoder(generated_tokens[:, :position + 1], new_encoder_out)
            predict_probs = decoder_output['output']
            predict_probs = self.log_probs(predict_probs)
            predict_probs[:, self.tgt_dict.pad()] = -math.inf
            predict_probs[:, self.tgt_dict.unk()] = -math.inf
            predict_probs[:, self.tgt_dict.bos()] = -math.inf
            
            if position >= self.max_length - 1:
                predict_probs[:] = -math.inf
                predict_probs[:, self.tgt_dict.eos()] = 0
                generated_tokens[:, position + 1] = self.tgt_dict.eos()
                break
            if position >= 1:
                eos_mask = generated_tokens[:, position].eq(self.tgt_dict.eos())
                # print("tokens at position ", position, ": ", generated_tokens[:, :position])
                # print(eos_mask)
                predict_probs[eos_mask] = -math.inf
                predict_probs[eos_mask, self.tgt_dict.eos()] = 0
            # predict_probs[finished_flag, self.tgt_dict.pad()] = math.inf
            # print(predict_probs[finished_flag, self.tgt_dict.pad()])
            candidate_scores, candidate_idxs, candidate_beam_idxs = self.beam_search(predict_probs, scores.view(batch_size, self.beam_size, -1), generated_tokens, position)


            cand_bbsz_idx = candidate_beam_idxs + bbsz_offsets    ## batch， beam, 对应到generated token中的位置
            cand_bbsz_idx = cand_bbsz_idx.view(-1)


            generated_tokens = generated_tokens[cand_bbsz_idx]
            generated_tokens[:, position + 1] = candidate_idxs

            # generated_tokens[cand_bbsz_idx, position + 1] = candidate_idxs

            # print("beam search words: ", candidate_idxs)
            # print("beam search beam idx:", cand_bbsz_idx)
            # print("updated sequence: ")
            # print(generated_tokens[:, :position + 2])
            # pause = input("??")

            scores[:, position + 1] = candidate_scores
            final_scores = scores[:, position + 1]
            # final_scores[new_eos_mask] = scores[new_eos_mask, position]  ## use the score before <eos>
            
            finished_flag = generated_tokens[:, position + 1].eq(self.tgt_dict.eos())
            
            if finished_flag.sum() == batch_size * self.beam_size:
                break
            if position >= self.max_length:
                break
        
        batch_scores = final_scores.view(batch_size, self.beam_size)
        best_beam = torch.argmax(batch_scores, dim = -1)
        generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
        best_sentences = torch.stack([generated_tokens[i, best_beam[i]] for i in range(len(best_beam))], dim = 0)
        return best_sentences[:, 1:]

