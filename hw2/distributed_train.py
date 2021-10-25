import copy
import os

import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

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

from hw2.logging_module import create_logger

logger,_ = create_logger()


def train_epoch(model, dataloader, metric,optimizer, device):
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

def evaluate(model, dataloader, metric, device):
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



def train(model, dataset, rank, metric, valid_dataset):
    print(f"Running on rank {rank}.")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(13453)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=2)

    model = copy.deepcopy(model).to(rank)
    model = DDP(model, device_ids=[rank])


    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas = 2, rank = rank)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

    train_loader = DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = 0, collate_fn = dataset.collater, sampler = train_sampler)
    # valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False, num_workers = 0, collate_fn = dataset.collater, sampler = valid_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False, num_workers = 0, collate_fn = dataset.collater)

    model = model.to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epoch_iter = 100
    patience = 0
    max_score = 0
    min_loss = np.inf
    best_model = None
    for epoch in range(epoch_iter):
        train_epoch(model, train_loader, metric, optimizer, rank)
        if rank == 0:
            train_loss, train_acc = evaluate(model, valid_loader, metric , rank)

            valid_loss, valid_acc = evaluate(model, valid_loader, metric , rank)

            message = f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {valid_acc:.4f}"
            print(message)
            logger.info(message)
            
            object_list = [val_loss, valid_acc]
        else:
            object_list = [None, None]
        dist.broadcast_object_list(object_list, src=0)
        val_loss, valid_acc = object_list

        if val_loss <= min_loss or valid_acc >= max_score:
            if val_loss <= min_loss:
                best_model = copy.deepcopy(model)
            min_loss = np.min((min_loss, val_loss))
            max_score = np.max((max_score, valid_acc))
            patience = 0

        dist.barrier()

        if rank == 0:
            torch.save(best_model, "model.pt")

    dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':

    world_size = 2

    mp.set_start_method("spawn", force=True)
    # mp.set_sharing_strategy('file_system')

    en_train_corpus_dir = '/data/bairu/mt_dataset/opus/en.BPE'
    ha_train_corpus_dir = '/data/bairu/mt_dataset/opus/ha.BPE'

    en_valid_corpus_dir = '/data/bairu/mt_dataset/opus/en_dev.BPE'
    ha_valid_corpus_dir = '/data/bairu/mt_dataset/opus/ha_dev.BPE'

    common_dict = dictionary()
    common_dict.build_vocab([en_train_corpus_dir, ha_train_corpus_dir])
    print("vocab size: ", len(common_dict.word_list))

    model_config = BairuConfig()
    token_embedding = torch.nn.Embedding(num_embeddings = common_dict.num_tokens, embedding_dim = model_config.embedding_dim, padding_idx = common_dict.pad())
    encoder = BairuTransformerEncoder(model_config, common_dict)
    decoder = BairuLSTMDecoder(model_config, common_dict)
    seq2seq_model = BairuEncoderDecoderModel(encoder, decoder)

    metric = CrossEntropyLossMetric(data_dict = common_dict, debug = False)
    optimizer = torch.optim.AdamW(seq2seq_model.parameters(), lr = 1e-4)

    train_dataset = MTDataset(src_dict = common_dict, tgt_dict = common_dict, src_corpus_dir = en_train_corpus_dir, tgt_corpus_dir = ha_train_corpus_dir,
                                max_len = 100, sanity_check = False)

    valid_dataset = MTDataset(src_dict = common_dict, tgt_dict = common_dict, src_corpus_dir = en_valid_corpus_dir, tgt_corpus_dir = ha_valid_corpus_dir,
                                max_len = 100)


    device_count = torch.cuda.device_count()
    if device_count < world_size:
        size = device_count
        print(f"Available device count ({device_count}) is less than world size ({world_size})")
    else:
        size = world_size
    print(size)
    processes = []
    for rank in range(size):
        p = Process(target=train, args=(seq2seq_model, train_dataset, rank, metric, valid_dataset))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    model = torch.load("model.pt").to("cuda:0")
    loss, acc = evaluate(model, valid_dataset, metric, 0)



