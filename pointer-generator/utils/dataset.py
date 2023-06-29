# -*- coding: utf-8 -*-

import csv
import glob
import time
import queue
import struct
import numpy as np
import tensorflow as tf
from random import shuffle
from threading import Thread
from tensorflow.core.example import example_pb2

from utils import utils
from utils import config

import random
random.seed(1234)


# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_STA = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
BOS_TOKEN = '[BOS]'  # This has a vocab id, which is used at the start of every decoder input sequence
EOS_TOKEN = '[EOS]'  # This has a vocab id, which is used at the end of untruncated target sequences
# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):

    def __init__(self, file, max_size):
        self.word2idx = {}
        self.idx2word = {}
        self.count = 0     # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [BOS] and [EOS] get the ids 0,1,2,3.
        for w in [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
            self.word2idx[w] = self.count
            self.idx2word[self.count] = w
            self.count += 1

        # Read the vocab file and add words up to max_size
        sentences = []
        with open(file, 'r') as fin:
            for line in fin:
                items = line.split()
                if len(items) != 2:
                    items = ['_'.join(items[:-1]), items[-1]]
                    # print('Warning: incorrectly formatted line in vocabulary file: %s' % line.strip())
                    continue
                w = items[0]
                if w in [SENTENCE_STA, SENTENCE_END, UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [BOS] and [EOS] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self.word2idx:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self.word2idx[w] = self.count
                self.idx2word[self.count] = w
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
          self.count, self.idx2word[self.count - 1]))
        
        if config.use_idf:
            idf_float_dict = {}
            with open(config.idf_path, 'r') as f:
                for line in f:
                    items = line.strip().split()
                    idf_float_dict[items[0]] = float(items[1])
            values = list(idf_float_dict.values())
            minidf = min(values)
            maxidf = max(values)
            self.idf_dict = {}
            self.idf_dict[UNK_TOKEN] = 0
            for w in list(idf_float_dict.keys()):
                if w in self.word2idx:
                    self.idf_dict[w] = int(config.num_idf_bins*(idf_float_dict[w] - minidf)/(maxidf - minidf))
                    if self.idf_dict[w] == config.num_idf_bins:
                        self.idf_dict[w]-=1
        
    def word2idfid(self, word):
        if not config.use_idf:
            return None
        if word not in self.word2idx or word not in self.idf_dict:
            return self.idf_dict[UNK_TOKEN]
        return self.idf_dict[word]
        
    def word2id(self, word):
        if word not in self.word2idx:
            return self.word2idx[UNK_TOKEN]
        return self.word2idx[word]

    def id2word(self, word_id):
        if word_id not in self.idx2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.idx2word[word_id]

    def size(self):
        return self.count

    def write_metadata(self, path):
        print( "Writing word embedding metadata file to %s..." % (path))
        with open(path, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self.idx2word[i]})
import underthesea
class Example(object):

    def __init__(self, article, abstract, vocab, vocab_pos=None, pos=None, vocab_ner=None, ner=None):
        # Get ids of special tokens
        bos_decoding = vocab.word2id(BOS_TOKEN)
        eos_decoding = vocab.word2id(EOS_TOKEN)

        # Process the article
        if config.use_pos:
            article_words = []
            for sent in underthesea.sent_tokenize(article.decode()):
                article_words.extend(underthesea.word_tokenize(sent))
        else:
            article_words = [w.decode() for w in article.split()]
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        self.enc_inp = [vocab.word2id(w) for w in
                          article_words]   # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        # abstract = ' '.encode().join(abstract_sentences).decode()
        if config.use_pos:
            abstract_words = []
            for sent in underthesea.sent_tokenize(abstract.decode()):
                abstract_words.extend(underthesea.word_tokenize(sent))
        else:
            abstract_words = [w.decode() for w in abstract.split()]
        abs_ids = [vocab.word2id(w) for w in
                   abstract_words]         # list of word ids; OOVs are represented by the id for UNK token

        if config.use_pos:
            pos_tags = pos.split()  # list of strings
            if len(pos_tags) > config.max_enc_steps:
                pos_tags = pos_tags[:config.max_enc_steps]
            self.pos_ids = [vocab_pos.word2id(w) for w in
                    pos_tags]         # list of word ids; OOVs are represented by the id for UNK token
        if config.use_ner:
            ner_tags = ner.split()  # list of strings
            if len(ner_tags) > config.max_enc_steps:
                ner_tags = ner_tags[:config.max_enc_steps]
            self.ner_ids = [vocab_ner.word2id(w) for w in
                    ner_tags]         # list of word ids; OOVs are represented by the id for UNK token
        if config.use_idf:
            self.idf = [vocab.word2idfid(w) for w in article_words]
        
        # Get the decoder input sequence and target sequence
        self.dec_inp, self.tgt = self.get_dec_seq(abs_ids, config.max_dec_steps, bos_decoding, eos_decoding)
        self.dec_len = len(self.dec_inp)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id;
            # also store the in-article OOVs words themselves
            self.enc_inp_extend_vocab, self.article_oovs = utils.article2ids(article_words, vocab)
            if config.use_pos:
                self.enc_inp_extend_vocab_pos, self.pos_oovs = utils.article2ids(pos_tags, vocab_pos)
            if config.use_ner:
                self.enc_inp_extend_vocab_ner, self.ner_oovs = utils.article2ids(ner_tags, vocab_ner)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = utils.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.tgt = self.get_dec_seq(abs_ids_extend_vocab, config.max_dec_steps, bos_decoding, eos_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_pos = pos
        self.original_ner = ner

    def get_dec_seq(self, sequence, max_len, start_id, stop_id):
        src = [start_id] + sequence[:]
        tgt = sequence[:]
        if len(src) > max_len:   # truncate
            src = src[:max_len]
            tgt = tgt[:max_len]  # no end_token
        else:  # no truncation
            tgt.append(stop_id)  # end token
        assert len(src) == len(tgt)
        return src, tgt

    def pad_enc_seq(self, max_len, pad_id):
        while len(self.enc_inp) < max_len:
            self.enc_inp.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_inp_extend_vocab) < max_len:
                self.enc_inp_extend_vocab.append(pad_id)

    def pad_enc_idf_seq(self, max_len, pad_id):
        while len(self.idf) < max_len:
            self.idf.append(pad_id)

    def pad_enc_pos_seq(self, max_len, pad_id):
        while len(self.pos_ids) < max_len:
            self.pos_ids.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_inp_extend_vocab_pos) < max_len:
                self.enc_inp_extend_vocab_pos.append(pad_id)

    def pad_enc_ner_seq(self, max_len, pad_id):
        while len(self.ner_ids) < max_len:
            self.ner_ids.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_inp_extend_vocab_ner) < max_len:
                self.enc_inp_extend_vocab_ner.append(pad_id)

    def pad_dec_seq(self, max_len, pad_id):
        while len(self.dec_inp) < max_len:
            self.dec_inp.append(pad_id)
        while len(self.tgt) < max_len:
            self.tgt.append(pad_id)


class Batch(object):
    def __init__(self, example_list, vocab, batch_size, vocab_pos=None, vocab_ner=None):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(PAD_TOKEN)  # id of the PAD token used to pad sequences
        self._vocab_pos = None
        self._vocab_ner = None
        if vocab_pos is not None:
            self._vocab_pos = vocab_pos
            self.pad_pos_id = vocab_pos.word2id(PAD_TOKEN)  # id of the PAD token used to pad sequences
        if vocab_ner is not None:
            self._vocab_ner = vocab_ner
            self.pad_ner_id = vocab_ner.word2id(PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])
        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_enc_seq(max_enc_seq_len, self.pad_id)
            if config.use_pos:
                ex.pad_enc_pos_seq(max_enc_seq_len, self.pad_pos_id)
            if config.use_ner:
                ex.pad_enc_ner_seq(max_enc_seq_len, self.pad_ner_id)
            if config.use_idf:
                ex.pad_enc_idf_seq(max_enc_seq_len, 0)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        if config.use_pos:
            self.enc_pos_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        if config.use_ner:
            self.enc_ner_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        if config.use_idf:
            self.enc_idf_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_inp[:]
            if config.use_pos:
                self.enc_pos_batch[i, :] = ex.pos_ids[:]        
            if config.use_ner:
                self.enc_ner_batch[i, :] = ex.ner_ids[:]   
            if config.use_idf:
                self.enc_idf_batch[i] = ex.idf     
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            if config.use_pos:
                self.pos_oovs = [ex.pos_oovs for ex in example_list]
            if config.use_ner:
                self.ner_oovs = [ex.ner_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            if config.use_pos:
                self.enc_pos_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            if config.use_ner:
                self.enc_ner_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_inp_extend_vocab[:]
                if config.use_pos:
                    self.enc_pos_batch_extend_vocab[i, :] = ex.enc_inp_extend_vocab_pos[:]
                if config.use_ner:
                    self.enc_ner_batch_extend_vocab[i, :] = ex.enc_inp_extend_vocab_ner[:]

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_dec_seq(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.tgt_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_inp[:]
            self.tgt_batch[i, :] = ex.tgt[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_pos = [ex.original_pos for ex in example_list]  # list of list of lists


class Batcher(object):
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, vocab, data_path, batch_size, single_pass, mode, vocab_pos=None, vocab_ner=None):
        self._vocab = vocab
        self._vocab_pos = vocab_pos
        self._vocab_ner = vocab_ner
        self._data_path = data_path
        self.batch_size = batch_size
        self.single_pass = single_pass
        self.mode = mode

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1    # just one thread to batch examples
            self._bucketing_cache_size = 1   # only load one batch's worth of examples before bucketing
            self._finished_reading = False   # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 1  # num threads to fill example queue
            self._num_batch_q_threads = 1    # num threads to fill batch queue
            self._bucketing_cache_size = 1   # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:                   # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.compat.v1.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self.single_pass and self._finished_reading:
                tf.compat.v1.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        example_generator = self.example_generator(self._data_path, self.single_pass)
        input_gen = self.pair_generator(example_generator)

        while True:
            try:
                if config.use_pos and config.use_ner:
                    (article,
                    abstract,
                    pos, ner) = input_gen.__next__()  
                elif config.use_pos:
                    (article,
                    abstract,
                    pos) = input_gen.__next__() 
                elif config.use_ner:
                    (article,
                    abstract,
                    ner) = input_gen.__next__() 
                else:  
                    (article,
                    abstract,) = input_gen.__next__() 
            except StopIteration:  # if there are no more examples:
                tf.compat.v1.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self.single_pass:
                    tf.compat.v1.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            abstract_sentences = [sent.strip() for sent in utils.abstract2sents(
                abstract)]  # Use the <s> and </s> tags in abstract to get a list of sentences.
            if config.use_pos and config.use_ner:
                example = Example(article, abstract_sentences[0], self._vocab, self._vocab_pos, pos, self._vocab_ner, ner)
            elif config.use_pos:
                example = Example(article, abstract_sentences[0], self._vocab, self._vocab_pos, pos)
            elif config.use_ner:
                example = Example(article, abstract_sentences[0], self._vocab, self._vocab_ner, ner)
            else:
                example = Example(article, abstract_sentences[0], self._vocab)
            add = False
            if config.use_pos and config.use_ner:
                if len(example.enc_inp) == len(example.pos_ids) and len(example.enc_inp) == len(example.ner_ids):
                    add = True
            elif config.use_pos:
                if len(example.enc_inp) == len(example.pos_ids):
                    add = True
            elif config.use_ner:
                if len(example.enc_inp) == len(example.ner_ids):
                    add = True
            else:
                add = True
            
            if add:
                self._example_queue.put(example)
            else:
                print(1111)

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                if config.use_pos and config.use_ner:
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size, self._vocab_pos, self._vocab_ner))
                elif config.use_pos:
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size, self._vocab_pos))
                elif config.use_ner:
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size, None, self._vocab_ner))
                else:
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self.single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    if config.use_pos and config.use_ner:
                        self._batch_queue.put(Batch(b, self._vocab, self.batch_size, self._vocab_pos, self._vocab_ner))
                    elif config.use_pos:
                        self._batch_queue.put(Batch(b, self._vocab, self.batch_size, self._vocab_pos))
                    elif config.use_ner:
                        self._batch_queue.put(Batch(b, self._vocab, self.batch_size, None, self._vocab_ner))
                    else:
                        self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

    def watch_threads(self):
        while True:
            tf.compat.v1.logging.info(
                'Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.compat.v1.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.compat.v1.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def pair_generator(self, example_generator):
        while True:
            e = example_generator.__next__()  # e is a tf.Example
            try:
                article_text = e.features.feature['article'].bytes_list.value[
                    0]  # the article text was saved under the key 'article' in the data files
                abstract_text = e.features.feature['abstract'].bytes_list.value[
                    0]  # the abstract text was saved under the key 'abstract' in the data files
                if config.use_pos:
                    pos = e.features.feature['pos_tag'].bytes_list.value[0]  # the abstract text was saved under the key 'abstract' in the data files
                if config.use_ner:
                    ner = e.features.feature['ner_tag'].bytes_list.value[0]  # the abstract text was saved under the key 'abstract' in the data files
            except ValueError:
                tf.compat.v1.logging.error('Failed to get article or abstract from example')
                continue
            if len(article_text) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                # tf.compat.v1.logging.warning('Found an example with empty article text. Skipping it.')
                continue
            else:
                if config.use_pos and config.use_ner:
                    yield (article_text, abstract_text, pos, ner)
                elif config.use_pos:
                    yield (article_text, abstract_text, pos)
                elif config.use_ner:
                    yield (article_text, abstract_text, ner)
                else:
                    yield (article_text, abstract_text)

    def example_generator(self, data_path, single_pass):
        while True:
            filelist = glob.glob(data_path)  # get the list of datafiles
            assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
            if single_pass:
                filelist = sorted(filelist)
            else:
                random.shuffle(filelist)
            for f in filelist:
                reader = open(f, 'rb')
                while True:
                    len_bytes = reader.read(8)
                    if not len_bytes: break  # finished reading this file
                    str_len = struct.unpack('q', len_bytes)[0]
                    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                    yield example_pb2.Example.FromString(example_str)
            if single_pass:
                print("example_generator completed reading all datafiles. No more data.")
                break