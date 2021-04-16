# from linguistic_style_transfer_pytorch.config import GeneralConfig
import json
import os
import numpy as np
import collections
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from gensim.models import KeyedVectors

ROOT = os.getcwd()

# Part of Code taken from https://github.com/vineetjohn/linguistic-style-transfer/tree/master/linguistic_style_transfer_model/utils

class GeneralConfig:
    """
    General configuration
    """

    def __init__(self):
        # original vocab size
        self.vocab_size = 9200
        self.bow_hidden_dim = 7526
        self.data_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data")
        self.vocab_save_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data")
        self.train_pos_reviews_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "raw", "sentiment.train.1.txt")
        self.train_neg_reviews_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "raw", "sentiment.train.0.txt")
        self.train_text_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "clean", "yelp_train_data.txt")
        self.train_labels_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "clean", "yelp_train_labels.txt")
        self.pos_sentiment_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "lexicon", "positive-words.txt")
        self.neg_sentiment_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "lexicon", "negative-words.txt")
        self.word_embedding_text_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "embedding.txt")
        self.word_embedding_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "word_embeddings.npy")
        self.w2i_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "word2index.json")
        self.i2w_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "index2word.json")
        self.bow_file_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "data", "bow.json")
        self.model_save_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "checkpoints")
        self.avg_style_emb_path = os.path.join(
            ROOT, "linguistic_style_transfer_pytorch", "checkpoints", "avg_style_emb.pkl")
        self.embedding_size = 300
        self.pad_token = 0
        self.sos_token = 1
        self.unk_token = 2
        self.predefined_word_index = {
            "<pad>": 0,
            "<sos>": 1,
            "<unk>": 2,
        }
        self.filter_sentiment_words = True
        self.filter_stopwords = True


class Vocab:
    """
    Holds all the necessary methods to create vocabulary
    """

    def __init__(self, config):

        self.config = config
        self.vocab_size = config.vocab_size
        self.train_file_path = config.train_text_file_path
        self.vocab_save_path = config.vocab_save_path
        self.predefined_word_index = config.predefined_word_index
        self.filter_sentiment_words = config.filter_sentiment_words
        self.filter_stopwords = config.filter_stopwords

    def create_vocab(self):
        """
        Creates word2index and index2word dictionaries
        """
        index2word = dict()
        words = collections.Counter()
        word2index = self.predefined_word_index

        with open(self.train_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if len(line) == 0:
                    continue
                words.update(line.split())
        # Store only 9200 most common words
        words = words.most_common(self.vocab_size)
        logging.debug("collected {} most common words".format(self.vocab_size))
        # create embedding matrix
        emb_matrix = np.zeros(
            (self.config.vocab_size+len(self.config.predefined_word_index), self.config.embedding_size))
        # randomly initialize the special tokens
        emb_matrix[:len(self.config.predefined_word_index), :] = np.random.rand(
            len(self.config.predefined_word_index), self.config.embedding_size)
        # load the pretrained word embeddings
        w2v_model = KeyedVectors.load_word2vec_format(
            self.config.word_embedding_text_file_path).wv

        # fill the dictionary with special tokens first
        idx = 0
        for word, index in self.config.predefined_word_index.items():
            word2index[word] = index
            index2word[index] = word
            idx += 1
        # Create word2index, index2word by iterating over
        # the most common words
        for token in words:
            if token[0] in w2v_model:
                word2index[token[0]] = idx
                index2word[idx] = token[0]
                emb_matrix[idx, :] = w2v_model[token[0]]
                idx = idx + 1
        print("Created embeddings")
        print("Created word2index dictionary")
        print("Created index2word dictionary")
        # Saving the vocab file
        with open('./word2index.json', 'w') as json_file:
            json.dump(word2index, json_file)
        print("Saved word2index.json at {}".format(
            self.vocab_save_path+'/word2index.json'))
        with open('./index2word.json', 'w') as json_file:
            json.dump(index2word, json_file)
        print("Saved index2word.json at {}".format(
            self.vocab_save_path+'/index2word.json'))
        # Save word embedding file
        np.save('./word_embeddings.npy', emb_matrix)
        # create bow vocab
        self._populate_word_blacklist(word2index)

    def _populate_word_blacklist(self, word_index):
        """
        Creates a dict of vocab indeces of non-stopwords and non-sentiment words
        """
        blacklisted_words = set()
        bow_filtered_vocab_indices = dict()
        # The '|' operator on sets in python acts as a union operator
        blacklisted_words |= set(self.predefined_word_index.values())
        if self.filter_sentiment_words:
            blacklisted_words |= self._get_sentiment_words()
        if self.filter_stopwords:
            blacklisted_words |= self._get_stopwords()

        allowed_vocab = word_index.keys() - blacklisted_words
        i = 0

        for word in allowed_vocab:
            vocab_index = word_index[word]
            bow_filtered_vocab_indices[vocab_index] = i
            i += 1

        self.config.bow_size = len(allowed_vocab)
        print("Created word index blacklist for BoW")
        print("BoW size: {}".format(self.config.bow_size))
        # saving bow vocab
        with open('./bow.json', 'w') as json_file:
            json.dump(bow_filtered_vocab_indices, json_file)
        print("Saved bow.json at {}".format(
            self.vocab_save_path+'/bow.json'))

    def _get_sentiment_words(self):
        """
        Returns all the sentiment words (positive and negative)
        which are excluded from the main vocab to form the BoW vocab
        """
        with open(file=config.pos_sentiment_file_path,
                  mode='r', encoding='ISO-8859-1') as pos_sentiment_words_file,\
            open(file=config.neg_sentiment_file_path,
                 mode='r', encoding='ISO-8859-1') as neg_sentiment_words_file:
            pos_words = pos_sentiment_words_file.readlines()
            neg_words = neg_sentiment_words_file.readlines()
            words = pos_words + neg_words
        words = set(word.strip() for word in words)

        return words

    def _get_stopwords(self):
        """
        Returns all the stopwords which excluded from the
        main vocab to form the BoW vocab
        """
        nltk_stopwords = set(stopwords.words('english'))
        sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

        all_stopwords = set()
        # The '|' operator on sets in python acts as a union operator
        all_stopwords |= spacy_stopwords
        all_stopwords |= nltk_stopwords
        all_stopwords |= sklearn_stopwords

        return all_stopwords


if __name__ == "__main__":
    config = GeneralConfig()
    # self.train_text_file_path = os.path.join(
            # ROOT, "linguistic_style_transfer_pytorch", "data", "clean", "yelp_train_data.txt")
    config.train_text_file_path = "../data/clean/yelp_train_data.txt"
    config.word_embedding_text_file_path = "../data/embedding.txt"
    config.pos_sentiment_file_path = "../data/lexicon/positive-words.txt"
    config.neg_sentiment_file_path = "../data/lexicon/negative-words.txt"
    vocab = Vocab(config)
    vocab.create_vocab()
