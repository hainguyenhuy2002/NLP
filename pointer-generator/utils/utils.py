# -*- coding: utf-8 -*-

import os
import pyrouge
import logging
import numpy as np

import torch
import tensorflow as tf
from torch.autograd import Variable

from utils import config
from tensorflow.core.example import example_pb2
import pandas as pd
import underthesea
import struct
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pyvi import ViTokenizer


def article2ids(article_words, vocab):
    ids = []
    oov = []
    unk_id = vocab.word2id(config.UNK_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oov:  # Add to list of OOVs
                oov.append(w)
            oov_num = oov.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oov


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(config.UNK_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, \
                "Error: models produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: models produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur_p = 0
    sents = []
    while True:
        try:
            sta_p = abstract.index(config.SENTENCE_STA.encode(), cur_p)
            end_p = abstract.index(config.SENTENCE_END.encode(), sta_p + 1)
            cur_p = end_p + len(config.SENTENCE_END.encode())
            sents.append(abstract[sta_p + len(config.SENTENCE_STA.encode()):end_p])
        except ValueError as e:  # no more sentences
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(config.UNK_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(config.UNK_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str


def print_results(article, abstract, decoded_output):
    print("")
    print('ARTICLE:  %s', article)
    print('REFERENCE SUMMARY: %s', abstract)
    print('GENERATED SUMMARY: %s', decoded_output)
    print("")


def make_html_safe(s):
    if type(s) == bytes:
        s = s.decode()
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)

from torchmetrics.text.rouge import ROUGEScore
from glob import glob
from tqdm import tqdm
def rouge_eval_1(preds_dir, tgts_dir, out_path):
    rouge = ROUGEScore()
    preds = glob(f'{preds_dir}/*')
    preds.sort()
    tgts = glob(f'{tgts_dir}/*')
    tgts.sort()
    assert len(preds) == len(tgts)
    
    f = open(out_path, 'w')
    f.writelines('rouge1_fmeasure,rouge1_precision,rouge1_recall,rouge2_fmeasure,rouge2_precision,rouge2_recall,rougeL_fmeasure,rougeL_precision,rougeL_recall,rougeLsum_fmeasure,rougeLsum_precision,rougeLsum_recall\n')
    for i in tqdm(range(len(preds))):
        pred_txt = open(preds[i], 'r').read().strip()
        tgt_txt = open(tgts[i], 'r').read().strip()
        score = list(rouge(pred_txt, tgt_txt).values())
        score_str = ','.join([str(p.item()) for p in score])
        f.writelines(score_str+'\n')
    f.close()

def rouge_log(results_dict, dir_to_write):
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s..." % (results_file))
    with open(results_file, "w") as f:
        f.write(log_str)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    # loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    with summary_writer.as_default():
        tf.summary.scalar(tag_name, running_avg_loss, step=step)
    summary_writer.flush()
    # loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    # summary_writer.add_summary(loss_sum, step)
    return running_avg_loss


def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]
        decoded_words = decoded_words[fst_period_idx + 1:]
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(reference_sents):
            f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

    # print("Wrote example %i to file" % ex_index)

def write_for_rouge_1(reference, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
    decoded_sents = ' '.join(decoded_words)
    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = make_html_safe(decoded_sents)
    reference_sents = make_html_safe(reference)

    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        f.write(reference_sents)
    with open(decoded_file, "w") as f:
        f.write(decoded_sents)


def get_input_from_batch(batch, use_cuda):
    extra_zeros = None
    enc_lens = batch.enc_lens
    max_enc_len = np.max(enc_lens)
    enc_batch_extend_vocab = None
    enc_pos_batch_extend_vocab = None
    enc_ner_batch_extend_vocab = None
    enc_pos_batch = None
    enc_ner_batch = None
    enc_idf_batch = None
    batch_size = len(batch.enc_lens)
    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    if config.use_pos:
        enc_pos_batch = Variable(torch.from_numpy(batch.enc_pos_batch).long())
    if config.use_ner:
        enc_ner_batch = Variable(torch.from_numpy(batch.enc_ner_batch).long())
    if config.use_idf:
        enc_idf_batch = Variable(torch.from_numpy(batch.enc_idf_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        if config.use_pos:
            enc_pos_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_pos_batch_extend_vocab).long())
        if config.use_ner:
            enc_ner_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_ner_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    c_t = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    enc_pos = np.zeros((batch_size, max_enc_len))
    for i, inst in enumerate(batch.enc_batch):
        for j, w_i in enumerate(inst):
            if w_i != config.PAD:
                enc_pos[i, j] = (j + 1)
            else:
                break
    enc_pos = Variable(torch.from_numpy(enc_pos).long())

    if use_cuda:
        c_t = c_t.cuda()
        enc_pos = enc_pos.cuda()
        enc_batch = enc_batch.cuda()
        if config.use_pos:
            enc_pos_batch = enc_pos_batch.cuda()
        if config.use_ner:
            enc_ner_batch = enc_ner_batch.cuda()
        if config.use_idf:
            enc_idf_batch = enc_idf_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()

        if enc_pos_batch_extend_vocab is not None:
            enc_pos_batch_extend_vocab = enc_pos_batch_extend_vocab.cuda()

    return enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, c_t, coverage, enc_pos_batch, enc_pos_batch_extend_vocab, enc_ner_batch, enc_ner_batch_extend_vocab, enc_idf_batch

import re
def cleaner(text: str):
    text = text.strip()
    text = text.lower()
    text = re.sub(r'(\d{1,})h(\d{1,})', r'\1 giờ \2 phút', text)
    text = re.sub(r'(\d{1,})/(\d{,2})', r'\1 tháng \2', text)
    text = re.sub(r'(\d{1,})/(\d{3,})', r'\1 năm \2', text)
    text = re.sub(r'[^\w\s._,]', ' ', text)
    text = re.sub(r'(\d)\.(\d)', r'\1\2', text)
    text = re.sub(r'\.{1,}', '.', text)
    text = re.sub(r'\,{1,}', ',', text)
    text = re.sub(r'(\d{1,})([^\W\d]{1,})', r'\1 \2', text)
    text = re.sub(r'([^\W\d]{1,})(\d{1,})', r'\1 \2', text)
    text = re.sub(r'(\s)(\_)(\w*)', r'\1\3', text)
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    words = text.split()
    text = ' '.join(words)
    return text

def get_input_from_text(text, vocab, use_cuda, vocab_pos=None):
    text = ViTokenizer.tokenize(text)
    text = cleaner(text)
    if config.use_pos:
        words = []
        pos_tags = []
        for sent in underthesea.sent_tokenize(text):
            words.extend(underthesea.word_tokenize(sent))
            pos_tags.extend([pos[-1] for pos in underthesea.pos_tag(sent)])
        pos_ids = [vocab_pos.word2id(w) for w in pos_tags]
        enc_pos = [pos_ids for _ in range(config.beam_size)]
        enc_pos_batch = Variable(torch.from_numpy(np.array(enc_pos)))
    else:
        words = text.split()
        enc_pos_batch=None
    if config.use_idf:
        idf = [vocab.word2idfid(w) for w in words]
        enc_idf = [idf for _ in range(config.beam_size)]
        enc_idf_batch = Variable(torch.from_numpy(np.array(enc_idf)))
    else:
        enc_idf_batch = None
    
    enc_inp = [vocab.word2id(w) for w in words]
    enc_inp = [enc_inp for _ in range(config.beam_size)]
    enc_batch = Variable(torch.from_numpy(np.array(enc_inp)))
    enc_len = [len(words) for _ in range(config.beam_size)]
    enc_padding_mask = torch.ones_like(enc_batch)
    enc_inp_extend_vocab, article_oov = article2ids(words, vocab)
    enc_inp_extend_vocab = [enc_inp_extend_vocab for _ in range(config.beam_size)]
    enc_batch_extend_vocab = Variable(torch.from_numpy(np.array(enc_inp_extend_vocab)))
    max_art_oov = len(article_oov)
    extra_zeros = None
    if max_art_oov > 0:
        extra_zeros = Variable(torch.zeros((config.beam_size, max_art_oov)))
    c_t = Variable(torch.zeros((config.beam_size, 2 * config.hidden_dim)))
    coverage = None
    if config.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if use_cuda:
        c_t = c_t.cuda()
        enc_batch = enc_batch.cuda()
        if config.use_pos:
            enc_pos_batch = enc_pos_batch.cuda()
        # if config.use_ner:
        #     enc_ner_batch = enc_ner_batch.cuda()
        if config.use_idf:
            enc_idf_batch = enc_idf_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if coverage is not None:
            coverage = coverage.cuda()
        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
               
    return enc_batch, enc_len, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, c_t, coverage, article_oov, enc_pos_batch, enc_idf_batch

def get_output_from_batch(batch, use_cuda):
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    batch_size = len(batch.dec_lens)
    dec_lens = Variable(torch.from_numpy(dec_lens)).float()
    tgt_batch = Variable(torch.from_numpy(batch.tgt_batch)).long()
    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()

    dec_pos = np.zeros((batch_size, config.max_dec_steps))
    for i, inst in enumerate(batch.dec_batch):
        for j, w_i in enumerate(inst):
            if w_i != config.PAD:
                dec_pos[i, j] = (j + 1)
            else:
                break
    dec_pos = Variable(torch.from_numpy(dec_pos).long())

    if use_cuda:
        dec_lens = dec_lens.cuda()
        tgt_batch = tgt_batch.cuda()
        dec_batch = dec_batch.cuda()
        dec_padding_mask = dec_padding_mask.cuda()
        dec_pos = dec_pos.cuda()

    return dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch

# def extract_tf_idf(vocab):
#     corpus = []
#     filelist = glob(config.train_data_path)  
#     assert filelist, ('Error: Empty filelist at %s' % config.train_data_path)  
#     filelist = sorted(filelist)
#     for f in tqdm(filelist):
#         reader = open(f, 'rb')
#         while True:
#             len_bytes = reader.read(8)
#             if not len_bytes: break  # finished reading this file
#             str_len = struct.unpack('q', len_bytes)[0]
#             example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
#             e = example_pb2.Example.FromString(example_str)
#             article_text = e.features.feature['article'].bytes_list.value[
#             0].decode()
#             abstract_text = abstract2sents(e.features.feature['abstract'].bytes_list.value[
#             0])[0].decode()
#             corpus.append(article_text)
#             corpus.append(abstract_text)

#     cv=CountVectorizer() 
#     word_count_vector=cv.fit_transform(corpus)
#     tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
#     tfidf_transformer.fit(word_count_vector)
#     max_idf = max(tfidf_transformer.idf_)
#     idx2tfidf = dict(zip([vocab.word2id(word) for word in cv.get_feature_names_out()], np.divide(tfidf_transformer.idf_, max_idf)))
#     return idx2tfidf

# def extract_pos_tag(vocab):
#     words = list(vocab.word2idx.values())
#     pos_tags = {}
#     for word in tqdm(words):
#         pos_tags[word] = underthesea.pos_tag(str(word))[0][1]
#         print(underthesea.pos_tag(str(word))[0][1])
#     df = pd.DataFrame(list(pos_tags.items()), columns=['word', 'pos'])
#     df['pos'] = pd.Categorical(df['pos'])
#     df['code'] = df['pos'].cat.codes
#     pos_dict ={}
#     for _, row in df.iterrows():
#        pos_dict[row['word']] = row['code']
#     return pos_dict

# def extract_ner_tag(vocab):
#     words = list(vocab.word2idx.values())
#     ner_tags = {}
#     for word in tqdm(words):
#         ner_tags[word] = underthesea.ner(str(word))[0][-1]
#     df = pd.DataFrame(list(ner_tags.items()), columns=['word', 'pos'])
#     df['pos'] = pd.Categorical(df['pos'])
#     df['code'] = df['pos'].cat.codes
#     ner_dict ={}
#     for _, row in df.iterrows():
#        ner_dict[row['word']] = row['code']
#     return ner_dict

