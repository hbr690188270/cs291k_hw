import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BairuEncoderDecoderModel
from .bairu_encoder import BairuEncoder
from .bairu_decoder import BairuDecoder
from ..data.bairu_dictionary import dictionary


class SequenceGeneratorFuck(nn.Module):
    def __init__(self, model: BairuEncoderDecoderModel, tgt_dict: dictionary, beam_size: int, max_length: int, beginning = 'eos'):
        self.model = model
        self.tgt_dict = tgt_dict
        self.beam_size = beam_size
        self.max_length = max_length
        if beginning == 'eos':   ## keep the same begining token with training phase
            self.begin_token = self.tgt_dict.eos()
        else:
            self.begin_token = self.tgt_dict.bos()

    def log_probs(self, logits):
        return F.log_softmax(logits, dim = -1)

    def beam_search(self, predict_prob, scores, generated_tokens, position):
        '''
        predict_prob: batch_size, vocab_num
        scores: batch_size, beam_size, max_length 

        add predict prob with prob score   first reshape the predict_prob into batch_size, beam_size, 1   then broadcast
        '''
        batch_size, vocab_num = scores.size()
        predict_prob = predict_prob.view(batch_size, 1, vocab_num).repeat(1, self.beam_size)
        predict_prob = predict_prob + scores[:,:, position - 1]    ## batch_size, beam_size, vocab_num

        top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size * 2)
        top_scores, top_indices = top_pred
        beam_idx = top_indices // vocab_num
        word_idx = torch.fmod(top_indices, vocab_num)
        return top_scores, word_idx, beam_idx

    def finalize_hypos(self, position,  end_batch_idx, end_batch_scores, generated_tokens, all_scores, finalized_list, finished):
        end_sentences = generated_tokens.index_select(0, end_batch_idx)[:, 1: position + 2]
        end_sentences[:, position] = self.tgt_dict.eos()
        end_scores = all_scores.index_select(0, end_batch_idx)[:,:position + 1]
        end_scores[:, position] = end_batch_scores
        end_scores[:, 1:] = end_scores[:, 1:] - end_scores[:, :-1]

        unfinished_list = []
        count = 0
        for flag in finished:
            if flag:
                count += 1
            else:
                unfinished_list.append(count)

        sents_seen = {}
        for i in range(end_batch_idx.size(0)):
            idx = end_batch_idx[i]
            score = end_batch_scores[i]
            unfinished_idx = idx // self.beam_size
            orig_sent_idx = unfinished_idx + unfinished_list[unfinished_idx]
            if orig_sent_idx not in sents_seen:
                sents_seen[orig_sent_idx] = None
            
            if len(finalized_list[unfinished_idx]) < self.beam_size:
                finalized_list[unfinished_idx].append(
                    {
                        "tokens": end_sentences[i],
                        "score": score,
                    }
                )
        newly_finished = []
        for orig_sent_idx in sents_seen.keys():
            pass


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
        encoder_hidden_dim = encoder_output.size(-1)

        ## expand the encoder_out to shape [batch_size * beam_size, seq_len, hidden_dim]
        new_encoder_out = encoder_out.view(batch_size, 1,seq_len, encoder_hidden_dim).repeat(1, self.beam_size)
        new_encoder_out = new_encoder_out.view(-1, seq_len, encoder_hidden_dim)
        
        ## start beam search
        generated_tokens = torch.zeros(batch_size * self.beam_size, self.max_length + 2).to(curr_device)
        batch_beam_offset = (torch.arange(0, batch_size,) * self.beam_size).unsqueeze(1).type_as(generated_tokens).to(curr_device)
        scores = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device)
        generated_tokens[0,:] = self.begin_token
        finished_samples = torch.zeros(batch_size * self.beam_size).to(curr_device).eq(-1)
        finished = [False for i in range(batch_size)]
        unfinished_sents = len(finished)

        for position in range(self.max_length):
            predict_probs, _ = self.model.decoder(generated_tokens[:, :position + 1])
            predict_probs = self.log_probs(predict_probs)
            predict_probs[:, self.tgt_dict.pad()] = -math.inf
            predict_probs[:, self.tgt_dict.unk()] = -math.inf
            
            if position >= self.max_length:
                predict_probs[:, self.tgt_dict.eos()] = math.inf
            candidate_scores, candidate_idxs, candidate_beam_idxs = self.beam_search(predict_probs, scores, generated_tokens, position)
            eos_mask = candidate_idxs.eq(self.tgt_dict.eos()) & candidate_scores.ne(-math.inf)
            eos_mask[:, :self.beam_size][finished_samples] = 0
            
            ## batch_size, beam_size;  batch_beam_offset: batch_size * 1
            candidate_batch_beam_idx = candidate_beam_idxs + batch_beam_offset

            eos_batch_beam_idx = torch.masked_select(candidate_batch_beam_idx[:, :self.beam_size], mask = eos_mask[:, :self.beam_size])
            if eos_batch_beam_idx.numel() > 0:
                eos_scores = torch.masked_select(candidate_scores, mask = eos_mask[:,:self.beam_size])
                finalized_sents = self.finalize_hypos()
                unfinished_sents -= len(finalized_sents)
            
            if unfinished_sents == 0:
                break
            if position >= self.max_length:
                break




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


class SequenceGeneratorFS(nn.Module):
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

        cands_to_ignore = (
            torch.zeros(batch_size, self.beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask


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

        generated_tokens[0,:] = self.begin_token
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
            
            if position >= self.max_length:
                predict_probs[:, self.tgt_dict.eos()] = math.inf
            print("applying finished mask: ", finished_flag)
            predict_probs[finished_flag, self.tgt_dict.pad()] = math.inf
            print(predict_probs[finished_flag, self.tgt_dict.pad()])
            candidate_scores, candidate_idxs, candidate_beam_idxs = self.beam_search(predict_probs, scores.view(batch_size, self.beam_size, -1), generated_tokens, position)

            cand_bbsz_idx = candidate_beam_idxs + bbsz_offsets    ## batch， beam, 对应到generated token中的位置
            eos_mask = candidate_idxs.eq(self.tgt_dict.eos) & candidate_scores.ne(-math.inf)        
            eos_mask[:, :self.beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :self.beam_size], mask=eos_mask[:, :self.beam_size]
            )

            # generated_tokens[cand_bbsz_idx] = candidate_idxs   但是有些generated tokens的序列不能被选，因为已经eos，所以需要先将这部分不能选的序列选出来
            
            new_eos_mask = (candidate_idxs.eq(self.tgt_dict.eos()) & (~finished_flag)).view(-1)
            finished_flag[new_eos_mask] = True

            generated_tokens[new_eos_mask, position + 1] = self.tgt_dict.eos()
            # print(candidate_idxs.size())
            generated_tokens[~finished_flag, position + 1] = candidate_idxs[~finished_flag]

            # eos_mask = candidate_idxs.eq(self.tgt_dict.eos())



            print(candidate_idxs)
            print(self.tgt_dict.eos())
            print(candidate_idxs.eq(self.tgt_dict.eos()))
            print((~finished_flag))
            print(new_eos_mask)
            print(finished_flag.sum())

            pause = input("??")


            scores[:, position + 1] = candidate_scores
            final_scores[new_eos_mask] = scores[new_eos_mask, position]  ## use the score before <eos>
            if finished_flag.sum() == batch_size * self.beam_size:
                break
            if position >= self.max_length:
                break
        
        batch_scores = final_scores.view(batch_size, self.beam_size)
        best_beam = torch.argmax(batch_scores, dim = -1)
        generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
        best_sentences = torch.stack([generated_tokens[i, best_beam[i]] for i in range(len(best_beam))], dim = 0)
        return best_sentences[:, 1:]

