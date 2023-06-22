import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import model
import pickle
import time
import numpy as np
import pandas as pd
import os
import json
import random
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import RobertaModel, RobertaTokenizer, AutoTokenizer
from utils import Recorder
from data_utils import ReRankingDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from model import RankingLoss
import math
from underthesea import word_tokenize

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)

# PATH = f"{os.path.dirname(os.path.realpath(__file__))}/"
PATH = os.path.dirname(os.path.realpath(__file__))

def base_setting(args):
    
    args.epoch = getattr(args, 'epoch', 1)
    args.report_freq = getattr(args, "report_freq", 100)
    args.accumulate_step = getattr(args, "accumulate_step", 12)
    args.margin = getattr(args, "margin", 0.01)
    args.gold_margin = getattr(args, "gold_margin", 0)
    args.model_type = getattr(args, "model_type", 'vinai/phobert-base-v2')
    args.warmup_steps = getattr(args, "warmup_steps", 10000)
    args.grad_norm = getattr(args, "grad_norm", 0)
    args.seed = getattr(args, "seed", 970903)
    args.no_gold = getattr(args, "no_gold", False)
    args.pretrained = getattr(args, "pretrained", None)
    args.max_lr = getattr(args, "max_lr", 2e-3)
    args.scale = getattr(args, "scale", 1)
    args.datatype = getattr(args, "datatype", "diverse")
    args.dataset = getattr(args, "dataset", "xsum")
    args.max_len = getattr(args, "max_len", 120)  # 120 for cnndm and 80 for xsum
    args.max_num = getattr(args, "max_num", 16)
    args.cand_weight = getattr(args, "cand_weight", 1)
    args.gold_weight = getattr(args, "gold_weight", 1)


def evaluation(args):
    # load data
    base_setting(args)
    tok = AutoTokenizer.from_pretrained(args.model_type)
    #start_idx = 128060
    #num_data = 22644
    #data/candidates/save_output
    num_data = args.num_test
    data_path = os.path.join(PATH, 'data', 'candidates', 'save_output')
    test_set = ReRankingDataset(data_path, args.model_type, start_idx = 128060 , num_data=num_data , is_test=True, maxlen=512, is_sorted=False, maxnum=args.max_num, is_untok=True, device=args.gpuid[0])
    dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=test_set.collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    scorer = model.ReRanker(model_path, tok.pad_token_id)
    if args.cuda:
        scorer = scorer.cuda()
    scorer = torch.load(os.path.join(PATH, "cache", args.model_pt),map_location=f'cuda:{args.gpuid[0]}')
    scorer.eval()
    model_name = args.model_pt.split("/")[0]

    def mkdir(path, force=False):
        if not os.path.exists(path):
            if force:
                os.makedirs(path)
            else:
                os.mkdir(path)

    print(model_name)
    model_result_path = os.path.join(PATH, 'result', model_name)
    model_reference_path = os.path.join(model_result_path, 'reference')
    model_candidate_path = os.path.join(model_result_path, 'candidate')

    mkdir(model_result_path, force=True)
    mkdir(model_reference_path)
    mkdir(model_candidate_path)

    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'])
    rouge1, rouge2, rougeLsum = 0, 0, 0
    cnt = 0
    acc = 0
    scores = []
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            print(f'batch {i}/{len(dataloader)}', end='\r')
            # if args.cuda:
            #     to_cuda(batch, args.gpuid[0])
            samples = batch["data"]
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], require_gold = False)
            similarity = output['score']
            similarity = similarity.cpu().numpy()
            max_ids = similarity.argmax(1)
            scores.extend(similarity.tolist())
            acc += (max_ids == batch["scores"].cpu().numpy().argmax(1)).sum()
            for j in range(similarity.shape[0]):
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                
                score = rouge_scorer.score(sample["abstract_untok"], sents)
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
                with open(os.path.join(model_candidate_path, f"{cnt}.dec"), "w", encoding='utf-8') as f:
                    print(sents, file = f)
                with open(os.path.join(model_reference_path, f"{cnt}.ref"), "w", encoding='utf-8') as f:
                    print(sample["abstract_untok"], file=f)
                cnt += 1

    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    print(f"accuracy: {acc / cnt}")
    print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))
    


def test(dataloader, scorer, args, gpuid):
    scorer.eval()
    loss = 0
    cnt = 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            # if args.cuda:
            #     to_cuda(batch, gpuid)
            samples = batch["data"]
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity.cpu().numpy()
            if i % 1000 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score(sample["abstract_untok"], sents)
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    scorer.train()
    loss = 1 - ((rouge1 + rouge2 + rougeLsum) / 3)
    print(f"rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")
    
    if len(args.gpuid) > 1:
        loss = torch.FloatTensor([loss]).to(gpuid)
        dist.all_reduce(loss, op=dist.reduce_op.SUM)
        loss = loss.item() / len(args.gpuid)
    return loss


def run(rank, args):
    
    base_setting(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    checkpoints_folder = os.path.join(PATH, "cache")
    if not os.path.exists(checkpoints_folder):
        os.mkdir(checkpoints_folder)
        
    if is_master:
        id = len(checkpoints_folder)
        recorder = Recorder(id, args.log)
    tok = AutoTokenizer.from_pretrained(args.model_type)
    print(f"GPU ID: {gpuid}, is mp: {is_mp}")
    #data/candidates/save_output
    #105418
    data_path = os.path.join(PATH, 'data', 'candidates', 'save_output')
    train_set = ReRankingDataset(data_path, args.model_type, start_idx = 0, num_data = args.num_train,maxlen=args.max_len, maxnum=args.max_num, device=gpuid)
    val_set = ReRankingDataset(data_path, args.model_type, start_idx =105418 , num_data = args.num_val,is_test=True, maxlen=512, is_sorted=False, maxnum=args.max_num, device=gpuid)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, collate_fn=train_set.collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=val_set.collate_fn, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=val_set.collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    scorer = model.ReRanker(model_path, tok.pad_token_id)
    #scorer = torch.compile(scorer)
    if len(args.model_pt) > 0:
        scorer = torch.load(os.path.join(checkpoints_folder, args.model_pt), map_location=f'cuda:{gpuid}')
    if args.cuda:
        if len(args.gpuid) == 1:
            scorer = scorer.cuda()
        else:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            scorer = nn.parallel.DistributedDataParallel(scorer.to(gpuid), [gpuid], find_unused_parameters=True)
    scorer.train()
    init_lr = args.max_lr / args.warmup_steps
    s_optimizer = optim.Adam(scorer.parameters(), lr=init_lr)
    if is_master:
        recorder.write_config(args, [scorer], __file__)
    minimum_loss = 100
    all_step_cnt = 0
    # start training
    for epoch in range(args.epoch):
        s_optimizer.zero_grad()
        step_cnt = 0
        sim_step = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if i % 10 == 0:
                print(f"step: {i}, loss: {avg_loss / (i + 1)}")
            # if args.cuda:
            #     to_cuda(batch, gpuid)
            step_cnt += 1
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['summary_score']
            loss = args.scale * RankingLoss(similarity, gold_similarity, args.margin, args.gold_margin, args.gold_weight)
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()
            if step_cnt == args.accumulate_step:
                # optimize step      
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)
                step_cnt = 0
                sim_step += 1
                all_step_cnt += 1
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()
            if sim_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                print("id: %d"%id)
                print(f"similarity: {similarity[:, :10]}")
                if not args.no_gold:
                    print(f"gold similarity: {gold_similarity}")
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f"%(epoch+1, sim_step, 
                 avg_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_loss = 0
            del similarity, gold_similarity, loss
            
            if  args.val_step is not None: 
                if all_step_cnt % args.val_step == 0 and all_step_cnt != 0 and step_cnt == 0:
                    loss = test(val_dataloader, scorer, args, gpuid)
                    if loss < minimum_loss and is_master:
                        minimum_loss = loss
                        if is_mp:
                            recorder.save(scorer.module, args.savepth)
                        else:
                            recorder.save(scorer, args.savepth)
                        recorder.save(s_optimizer, "optimizer.pth")
                        recorder.print("best - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                    if is_master:
                        recorder.print("val rouge: %.6f"%(1 - loss))
                    
        print(f"Complete epoch {epoch+1}")
    if args.val_step == None and is_master:
        recorder.save(scorer, args.savepth)
        print("Saved successfully")
               

def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        print("RUN ONE PROCESS")
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0)
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-p", "--port", type=int, default=12355)
    parser.add_argument("--model_pt", default="", type=str)
    parser.add_argument("--encode_mode", default=None, type=str)
    parser.add_argument("--num_train", default=105418, type=int)
    parser.add_argument("--num_val", default=22642, type=int)
    parser.add_argument("--num_test", default=22644, type=int)
    parser.add_argument("--val_step", default=None, type=int)
    parser.add_argument("--savepth", default="scorer.pth", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    
    args = parser.parse_args()
    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:    
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)