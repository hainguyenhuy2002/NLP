from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import torch
from functools import partial
import time
from transformers import RobertaTokenizer, AutoTokenizer
import random
import pickle
import copy
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# def to_cuda(batch, gpuid):
#     for n in batch:
#         if n != "data":
#             batch[n] = batch[n].to(gpuid)


class ReRankingDataset(Dataset):
    def __init__(self, fdir, model_type, * , start_idx, num_data, maxlen=-1, is_test=False, total_len=512, is_sorted=True, maxnum=-1, is_untok=True, device='cpu'):
        """ data format: article, abstract, [(candidiate_i, score_i)] """
        self.start_idx = start_idx
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            self.num = num_data
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            self.num = len(self.files)
        self.tok = AutoTokenizer.from_pretrained(model_type, verbose=False)
        self.maxlen = maxlen
        self.is_test = is_test
        self.pad_token_id = self.tok.pad_token_id
        self.total_len = total_len
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id
        self.sorted = is_sorted
        self.maxnum = maxnum
        self.is_untok = is_untok
        self.device = torch.device(device)

    def __len__(self):
        return self.num

    def bert_encode(self, x):
        ids = [self.cls_token_id]
        ids.extend(x)
        ids.append(self.sep_token_id)
        return ids

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%(idx+self.start_idx)), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)
        origin_text = data['article_untok']
        abstract_text = data["abstract_untok"]

        if self.sorted:
            data['candidates_untok'] = sorted(data['candidates_untok'], key=lambda x: x[1], reverse=True)

        candidates_text = [candidate[0] for candidate in data['candidates_untok']]
        scores = [candidate[1] for candidate in data['candidates_untok']]
        data['candidates'] = data.pop('candidates_untok')

        result = {
            "origin_text": origin_text,
            "abstract_text": abstract_text,
            "candidates_text": candidates_text,
            "scores": scores
        }
        
        if self.is_test:
            result["data"] = data

        # print(result)
        return result
    
    def collate_fn(self, batch):
        origin_text = [row['origin_text'] for row in batch]
        abstract_text = [row['abstract_text'] for row in batch]


        all_origin_ids = self.tok(origin_text, add_special_tokens=False, padding='longest')['input_ids']
        all_abstract_ids = self.tok(abstract_text, add_special_tokens=False, padding='longest')['input_ids']
        all_abstract_ids = [self.bert_encode(x) for x in all_abstract_ids]
        
        batch_size = len(batch)
        num_candidate = len(batch[0]['candidates_text'])

        all_candidates = [candidate for row in batch for candidate in row['candidates_text']]
        all_candidates_ids = self.tok(all_candidates, add_special_tokens=False, padding='longest')['input_ids']
        all_candidates_ids = [self.bert_encode(x) for x in all_candidates_ids]

        len_origin_ids = len(all_origin_ids[0])
        src_ids = []
        for i in range(0, len_origin_ids, 240):
            src_ids.append([])
            for origin_ids in all_origin_ids:
                src_ids[-1].append(self.bert_encode(origin_ids[i:i+240]))
            src_ids[-1] = torch.tensor(src_ids[-1], device=self.device, dtype=torch.long)

        all_abstract_ids = torch.tensor(all_abstract_ids, device=self.device, dtype=torch.long)

        scores = torch.tensor([row["scores"] for row in batch], device=self.device, dtype=torch.float)
        candidate_ids = torch.tensor(all_candidates_ids, device=self.device, dtype=torch.long).reshape(batch_size, num_candidate, -1)

        result = {
            "src_input_ids": src_ids,
            "tgt_input_ids": all_abstract_ids,
            "candidate_ids": candidate_ids,
            "scores": scores
        }
        
        if self.is_test:
            result["data"] = [row['data'] for row in batch]
            
        return result
