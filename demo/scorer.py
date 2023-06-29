import torch
from torch import nn
from transformers import RobertaModel, AutoTokenizer
from vncorenlp import VnCoreNLP
import py_vncorenlp

class ReRanker(nn.Module):
    def __init__(self, vncorenlp_path, device, pretrain_path='vinai/phobert-base-v2'):
        super(ReRanker, self).__init__()
        self.encoder = RobertaModel.from_pretrained(pretrain_path)
        self.tok = AutoTokenizer.from_pretrained(pretrain_path)
        self.pad_token_id = self.tok.pad_token_id
        self.segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)
        self.device = device

    def bert_encode(self, x):
        ids = [self.tok.cls_token_id]
        ids.extend(x)
        ids.append(self.tok.sep_token_id)
        return ids

    def forward(self, texts_id, candidate_id, summary_id=None, require_gold=True):
        
        batch_size = candidate_id.size(0)
        
        doc_embs = []
        mask_for_sim = []

        for text_id in texts_id:
            input_mask = text_id != self.pad_token_id
            doc_encoder_out = self.encoder(text_id, attention_mask=input_mask)[0]
            doc_embs.append(doc_encoder_out[:, 0, :])
            
            # because the first and last of text_id is cls and seq token -> 
            # if one row do not have any id != pad id, that row will be false (0)
            mask_for_sim.append(torch.any(input_mask[:, 1:-1], dim=1).int())

        mask_for_sim = torch.stack(mask_for_sim) # shape: num_divide x batch
        mask_for_sim = torch.permute(mask_for_sim, (1, 0)) # shape: batch x num_divide

        if require_gold:
            # get reference score
            input_mask = summary_id != self.pad_token_id
            out = self.encoder(summary_id, attention_mask=input_mask)[0]
            summary_emb = out[:, 0, :]
            summary_score = torch.stack(
                [torch.cosine_similarity(summary_emb, doc_emb, dim=-1) for doc_emb in doc_embs], 
                axis=1
            )
            summary_score = summary_score * mask_for_sim
            summary_score = torch.sum(summary_score, axis=1) / torch.sum(mask_for_sim, axis=1)

        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = candidate_id != self.pad_token_id

        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, -1)
        
        # get candidate score
        score = torch.stack([
            torch.cosine_similarity(candidate_emb, doc_emb.unsqueeze(1).expand_as(candidate_emb), dim=-1)
            for doc_emb in doc_embs
        ], dim=1)

        score = torch.permute(score, (0, 2, 1)) # batch x num_candidate x num_divide
        score = score * mask_for_sim.unsqueeze(1).expand_as(score)    
        score = torch.sum(score, dim=2)
        score = score / torch.sum(mask_for_sim, dim=1).unsqueeze(1).expand_as(score)
        output = {'score': score}
        if require_gold:
            output['summary_score'] = summary_score

        return output
    
    def get_best_candidate(self, origin_text: str, candidates_text: list):
        origin_text = [" ".join(self.segmenter.word_segment(origin_text))]
        all_candidates = [" ".join(self.segmenter.word_segment(candidate)) for candidate in candidates_text]

        all_origin_ids = self.tok(origin_text, add_special_tokens=False, padding='longest')['input_ids']
        batch_size = 1
        num_candidate = len(all_candidates)

        all_candidates_ids = self.tok(all_candidates, add_special_tokens=False, padding='longest')['input_ids']
        all_candidates_ids = [self.bert_encode(x) for x in all_candidates_ids]

        len_origin_ids = len(all_origin_ids[0])
        src_ids = []

        for i in range(0, len_origin_ids, 240):
            src_ids.append([])
            for origin_ids in all_origin_ids:
                src_ids[-1].append(self.bert_encode(origin_ids[i:i+240]))
            src_ids[-1] = torch.tensor(src_ids[-1], device=self.device, dtype=torch.long)

        candidate_ids = torch.tensor(all_candidates_ids, device=self.device, dtype=torch.long).reshape(batch_size, num_candidate, -1)

        with torch.no_grad():
            output = self(src_ids, candidate_ids, require_gold = False)
            similarity = output['score']
            similarity = similarity.cpu().numpy()
            
        max_ids = similarity.argmax(1)
        output = candidates_text[max_ids[0]]

        return output
            
