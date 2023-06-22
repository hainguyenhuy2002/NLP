# modified from https://github.com/maszhongming/MatchSum
import torch
from torch import nn
from transformers import RobertaModel


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class ReRanker(nn.Module):
    def __init__(self, encoder, pad_token_id):
        super(ReRanker, self).__init__()
        self.encoder = RobertaModel.from_pretrained(encoder)
        self.pad_token_id = pad_token_id

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