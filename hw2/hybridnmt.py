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
from hw2.models.utils import move_to_target_device

import tqdm
from hw2.logging_module import create_logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type = float, default = 1e-4)
parser.add_argument("--hidden_dropout", type = float, default = 0.1)
parser.add_argument("--hidden", type = int, default = 256)
parser.add_argument("--emb_dim", type = int, default = 256)
parser.add_argument("--layer", type = int, default = 2)
parser.add_argument("--ffn_dim", type = int, default = 1024)
parser.add_argument("--decay", type = float, default = 1e-6)
parser.add_argument("--att_dropout", type = float, default = 0.1)
parser.add_argument("--bs", type = int, default = 256)
parser.add_argument("--dec_init", type = str, default = "enc")

args = parser.parse_args()


logger,log_dir = create_logger()

device = torch.device("cuda")

en_train_corpus_dir = '/data/bairu/mt_dataset/opus/en.BPE'
ha_train_corpus_dir = '/data/bairu/mt_dataset/opus/ha.BPE'

en_valid_corpus_dir = '/data/bairu/mt_dataset/opus/en_dev.BPE'
ha_valid_corpus_dir = '/data/bairu/mt_dataset/opus/ha_dev.BPE'

common_dict = dictionary()
common_dict.build_vocab([en_train_corpus_dir, ha_train_corpus_dir])
print("vocab size: ", len(common_dict.word_list))

model_config = BairuConfig(embedding_dim = args.emb_dim, hidden_size = args.hidden, num_hidden_layers = args.layer, intermediate_size = args.ffn_dim, hidden_dropout_prob = args.hidden_dropout, 
                            decoder_embedding_dim = args.emb_dim, decoder_hidden_layer = args.layer, decoder_hidden_size = args.hidden,
                            attention_probs_dropout_prob = args.att_dropout, pad_token_id = common_dict.pad(), decoder_init = args.dec_init)

token_embedding = torch.nn.Embedding(num_embeddings = common_dict.num_tokens, embedding_dim = model_config.embedding_dim, padding_idx = common_dict.pad())
encoder = BairuTransformerEncoder(model_config, common_dict)
decoder = BairuLSTMDecoder(model_config, common_dict)
seq2seq_model = BairuEncoderDecoderModel(encoder, decoder).to(device)

metric = CrossEntropyLossMetric(data_dict = common_dict, debug = False)
optimizer = torch.optim.AdamW(seq2seq_model.parameters(), lr = args.lr, weight_decay = args.decay)

train_dataset = MTDataset(src_dict = common_dict, tgt_dict = common_dict, src_corpus_dir = en_train_corpus_dir, tgt_corpus_dir = ha_train_corpus_dir,
                            max_len = 100, sanity_check = False)
train_dataloader = DataLoader(train_dataset, batch_size = args.bs, shuffle = True, num_workers = 6, collate_fn = train_dataset.collater)

valid_dataset = MTDataset(src_dict = common_dict, tgt_dict = common_dict, src_corpus_dir = en_valid_corpus_dir, tgt_corpus_dir = ha_valid_corpus_dir,
                            max_len = 100)
valid_dataloader = DataLoader(valid_dataset, batch_size = args.bs, shuffle = False, num_workers = 6, collate_fn = train_dataset.collater)


# tqdm_train_dataloader = tqdm(train_dataloader)

def train_epoch(model, dataloader):
    loss_list = []
    total_correct = 0
    total_num = 0
    for i, batch_data in enumerate(tqdm.tqdm(dataloader)):
    # for batch_data in tqdm(dataloader):
        batch_data = move_to_target_device(batch_data, device)
        loss, correct, total = metric(model, batch_data,)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        total_correct += correct
        total_num += total
    return np.mean(loss_list), total_correct/total_num

def evaluate(model, dataloader):
    loss_list = []
    total_correct = 0
    total_num = 0
    for i, batch_data in enumerate(dataloader):
        batch_data = move_to_target_device(batch_data, device)
        with torch.no_grad():
            loss, correct, total = metric(model, batch_data,)
            loss_list.append(loss.item())
            total_correct += correct
            total_num += total
    return np.mean(loss_list), total_correct/total_num

best_eval_acc = 0
training_epoch = 100
for epoch in range(training_epoch):
    # train_loss, train_acc = train_epoch(seq2seq_model, tqdm_train_dataloader)
    train_loss, train_acc = train_epoch(seq2seq_model, train_dataloader)
    message = "epoch {}, train loss: {}, train acc: {}".format(epoch, train_loss, train_acc)
    print(message)
    logger.info(message)
    eval_loss, eval_acc = evaluate(seq2seq_model, valid_dataloader)
    message = "\t eval loss: {}, eval acc: {}\n".format(eval_loss, eval_acc)
    print(message)
    logger.info(message)
    # eval_loss, eval_acc = evaluate(seq2seq_model, valid_dataloader)
    if eval_acc > best_eval_acc:
        best_eval_acc = eval_acc
        torch.save(seq2seq_model, log_dir + '/model.pt')



