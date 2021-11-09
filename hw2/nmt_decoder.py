import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.utils.data import DataLoader

from hw2.data.bairu_dataset import MTDataset
from hw2.data.bairu_dictionary import dictionary
from hw2.models.transformer_encoder import BairuTransformerEncoder
from hw2.models.lstm_decoder import BairuLSTMDecoder
from hw2.models.base_model import BairuEncoderDecoderModel
from hw2.models.bairu_config import BairuConfig
from hw2.metric.cross_entropy_metric import CrossEntropyLossMetric
from sacrebleu.metrics import BLEU
from hw2.models.utils import move_to_target_device
from hw2.models.sequence_generator import SequenceGenerator
import hw2.models.utils as utils
from hw2.logging_module import create_logger
from preprocess_dataset import read_file, write_file
import tqdm
import os

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type = str, default = './logs/2021-11-08/07-58-03/model.pt')
parser.add_argument("--beam_size", type = int, default = 5)
parser.add_argument("--max_len", type = int, default = 50)
parser.add_argument("--bs", type = int, default = 1)
parser.add_argument("-i", type = str, default = 'aux_files/dev.en')
parser.add_argument("-eval", action = 'store_true')
parser.add_argument("--sanity_check", action = 'store_true')
parser.add_argument("--visualize", action = 'store_true')

args = parser.parse_args()

input_file = args.i

codes_file = './aux_files/bpe.txt'
en_vocab_file = './aux_files/en_vocab.txt'
ha_vocab_file = './aux_files/ha_vocab.txt'
en_output_file = './aux_files/bpe_en_test.txt'
ha_output_file = None

# print(input_file[])
if input_file[-4:] == '.xml':
    cmd = 'wmt-unwrap -o test_data < {}'.format(input_file)
    os.system(cmd)
    en_orig_test_file = './test_data.en'
    ha_orig_test_file = './test_data.ha'
    dev_en, en_errors = read_file(data_dir = './', filename = 'test_data.en')
    dev_ha, ha_errors = read_file(data_dir = './', filename = 'test_data.ha')
    all_errors = en_errors.union(ha_errors)
    new_dev_en = [dev_en[x] for x in range(len(dev_en)) if x not in all_errors]
    new_dev_ha = [dev_ha[x] for x in range(len(dev_ha)) if x not in all_errors]
    assert len(new_dev_en) == len(new_dev_ha)
    write_file(data_dir = './', data_list = new_dev_en, filename = 'dev_en-ha.txt.en')
    write_file(data_dir = './', data_list = new_dev_ha, filename = 'dev_en-ha.txt.ha')

    cmd = "subword-nmt apply-bpe -c {} --vocabulary {} --vocabulary-threshold 50 < {} > {}".format(codes_file, en_vocab_file,'./dev_en-ha.txt.en',en_output_file)
    os.system(cmd)

    ha_output_file = './aux_files/bpe_ha_test.txt'
    cmd = "subword-nmt apply-bpe -c {} --vocabulary {} --vocabulary-threshold 50 < {} > {}".format(codes_file, ha_vocab_file,'./dev_en-ha.txt.ha',ha_output_file)
    os.system(cmd)

else:
    dev_en, en_errors = read_file(data_dir = '', filename = input_file)
    write_file(data_dir = './', data_list = dev_en, filename = 'dev_en-ha.txt.en')

    cmd = "subword-nmt apply-bpe -c {} --vocabulary {} --vocabulary-threshold 50 < {} > {}".format(codes_file, en_vocab_file,'./dev_en-ha.txt.en',en_output_file)
    os.system(cmd)




# logger,log_dir = create_logger()

device = torch.device("cuda")

model = torch.load(args.model_dir)
model.eval()
common_dict = model.encoder.dictionary
print("vocab size: ", len(common_dict.word_list))

seq_generator = SequenceGenerator(model, tgt_dict = common_dict, beam_size = args.beam_size, max_length = args.max_len)

if args.eval:
    evaluate_bleu = True
else:
    evaluate_bleu = False

en_valid_corpus_dir = en_output_file
if ha_output_file is None:
    ha_valid_corpus_dir = en_output_file
    print("no reference file! will not evaluate BLEU score")
    evaluate_bleu = False
else:
    ha_valid_corpus_dir = ha_output_file

valid_dataset = MTDataset(src_dict = common_dict, tgt_dict = common_dict, src_corpus_dir = en_valid_corpus_dir, tgt_corpus_dir = ha_valid_corpus_dir,
                            max_len = 100, sanity_check = args.sanity_check, shuffle = False)

valid_dataloader = DataLoader(valid_dataset, batch_size = args.bs, shuffle = False, num_workers = 6, collate_fn = valid_dataset.collater)

if args.visualize:
    valid_dataloader = valid_dataloader
else:
    valid_dataloader = tqdm.tqdm(valid_dataloader)
hyp_list = []
ref_list = []
for batch_data in valid_dataloader:
    batch_data = move_to_target_device(batch_data, device)
    with torch.no_grad():
        gen_out = seq_generator(batch_data)
    # print(gen_out.size())
    for i in range(len(gen_out)):
        hyp = common_dict.decode(utils.strip_pad(gen_out[i], common_dict.pad(), common_dict.eos()).detach().cpu().numpy())
        if args.visualize:
            print("hyp: ", hyp)
        hyp_list.append(hyp)
        if evaluate_bleu:
            ref = common_dict.decode(utils.strip_pad(batch_data['net_input']['tgt_tokens'][i], common_dict.pad(), common_dict.eos()).detach().cpu().numpy())
            if args.visualize:
                print("ref: ", ref)
            ref_list.append([ref])
    if args.visualize:
        print()


## evaluation

with open("hyps_for_eval.txt",'w', encoding = 'utf-8') as f:
    for sentence in hyp_list:
        f.write(sentence + '\n')    

with open("refs_for_eval.txt",'w', encoding = 'utf-8') as f:
    for sentence in ref_list:
        f.write(sentence[0] + '\n')    

if evaluate_bleu:
    metric = BLEU()
    result = metric.corpus_score(hyp_list, ref_list)
    print("BLEU Score using API(a), the code API:\n\t", result)

    print("BLEU Score using API(b), the command line API: ")
    cmd = "sacrebleu refs_for_eval.txt -i hyps_for_eval.txt -m bleu -b -w 4"
    os.system(cmd)
    print("evaluation ends!")
else:
    with open("./decode_result.txt",'w', encoding = 'utf-8') as f:
        for sentence in hyp_list:
            f.write(sentence + '\n')

