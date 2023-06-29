from __future__ import unicode_literals, print_function, division

import os
import sys
import pandas as pd
import time
import torch
from torch.autograd import Variable

from models.model import Model
from utils import utils
from utils.dataset import Vocab
from utils import dataset, config
from utils.utils import get_input_from_text

use_cuda = config.use_gpu and torch.cuda.is_available()


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.state = state
        self.context = context
        self.coverage = coverage
        self.log_probs = log_probs

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.vocab_pos = None
        self.vocab_ner = None
        if config.use_pos:
            self.vocab_pos = Vocab(config.vocab_pos_path, 0)
        if config.use_ner:
            self.vocab_ner = Vocab(config.vocab_ner_path, 0)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True, is_tran=config.tran, vocab_pos_size=self.vocab_pos.size() if config.use_pos else None, vocab_ner_size=self.vocab_ner.size() if config.use_ner else None)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)
    

    def beam_search(self, text):
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, c_t, coverage, article_oov, enc_pos_batch, enc_idf_batch = get_input_from_text(text, self.vocab, use_cuda, self.vocab_pos)

        if not config.tran:
            # if config.use_pos and config.use_ner:
            #     enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens, enc_pos_batch, enc_ner_batch)
            if config.use_pos and config.use_idf:
                enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens, enc_pos_batch, None, enc_idf_batch)
            elif config.use_pos:
                enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens, enc_pos_batch)
            # elif config.use_ner:
            #     enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens, None, enc_ner_batch)
            elif config.use_idf:
                enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens, None, None, enc_idf_batch)
            else:
                enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens)
        # else:
            # enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_pos)
        s_t = self.model.reduce_state(enc_h)

        dec_h, dec_c = s_t  
        dec_h = dec_h.squeeze(0)
        dec_c = dec_c.squeeze(0)

        beams = [Beam(tokens=[self.vocab.word2id(config.BOS_TOKEN)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t[0],
                      coverage=(coverage[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]

        steps = 0
        results = []
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(config.UNK_TOKEN) \
                             for t in latest_tokens]
            y_t = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t = y_t.cuda()
            all_state_h = [h.state[0] for h in beams]
            all_state_c = [h.state[1] for h in beams]
            all_context = [h.context  for h in beams]

            s_t = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t = torch.stack(all_context, 0)

            coverage_t = None
            if config.is_coverage:
                all_coverage = [h.coverage for h in beams]
                coverage_t = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t, s_t,
                                                                                    enc_out, enc_fea,
                                                                                    enc_padding_mask, c_t,
                                                                                    extra_zeros, enc_batch_extend_vocab,
                                                                                    coverage_t, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(config.EOS_TOKEN):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0], article_oov
    
    def inference(self, text):
        start = time.time()
        best_summary, article_oov = self.beam_search(text)

        output_ids = [int(t) for t in best_summary.tokens[1:]]

        decoded_words = utils.outputids2words(output_ids, self.vocab,
                                                (article_oov if config.pointer_gen else None))

        try:
            fst_stop_idx = decoded_words.index(dataset.EOS_TOKEN)
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words

        print(f'Predict in {(time.time() - start)}:\n' + ' '.join(decoded_words).replace('_', ' '))

if __name__ == '__main__':
    model_file_path = sys.argv[1]
    text = sys.argv[2]
    test_processor = BeamSearch(model_file_path)
    test_processor.inference(text)