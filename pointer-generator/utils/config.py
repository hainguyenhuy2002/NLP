# -*- coding: utf-8 -*-

import os

SENTENCE_STA = '<s>'
SENTENCE_END = '</s>'

UNK = 0
PAD = 1
BOS = 2
EOS = 3

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'

beam_size=6
emb_dim= 128
batch_size= 8
hidden_dim= 512
max_enc_steps=800
max_dec_steps=50
max_tes_steps=50
min_dec_steps=10
vocab_size=100000

lr=0.15
cov_loss_wt = 1.0
pointer_gen = True
is_coverage = True

max_grad_norm=2.0
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4

eps = 1e-12
use_gpu=True
lr_coverage=0.15
max_iterations = 500000

# transformer
tran = False
d_k = 64
d_v = 64
n_head = 6
dropout = 0.1
n_layers = 6
d_model = 128
d_inner = 512
n_warmup_steps = 4000

use_pos = True
pos_dim = 32
use_ner = False
ner_dim = 64
use_posi = False
posi_dim = 4
use_idf = True
idf_dim = 32
num_idf_bins = 200
use_tf = False
tf_dim = 32

root_dir = os.path.expanduser("./")
log_root = os.path.join(root_dir, "dataset/log/")

train_data_path = os.path.join(root_dir, "dataset/finished_files_feat/chunked/train_*")
eval_data_path = os.path.join(root_dir, "dataset/finished_files_feat/val.bin")
decode_data_path = os.path.join(root_dir, "dataset/finished_files_feat/test.bin")
vocab_path = os.path.join(root_dir, "dataset/finished_files_feat/vocab")
vocab_pos_path = os.path.join(root_dir, "dataset/finished_files_feat/vocab_pos")
vocab_ner_path = os.path.join(root_dir, "dataset/finished_files_feat/vocab_ner")
train_position_path = os.path.join(root_dir, "dataset/train_position.npy")
val_position_path = os.path.join(root_dir, "dataset/val_position.npy")
test_position_path = os.path.join(root_dir, "dataset/test_position.npy")
idf_path = os.path.join(root_dir, "dataset/idf")
