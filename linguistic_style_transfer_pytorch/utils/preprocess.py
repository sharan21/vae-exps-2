import re
import logging
# from linguistic_style_transfer_pytorch.config import GeneralConfig
import os

ROOT = os.getcwd()


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

class Preprocessor():
    """
    Preprocessor class
    """

    def __init__(self):

        print("Preprocessor instantiated")

    def _clean_text(self, string):
        """
        Clean the raw text file
        """
        string = string.replace(".", "")
        string = string.replace(".", "")
        string = string.replace("\n", " ")
        string = string.replace(" 's", " is")
        string = string.replace("'m", " am")
        string = string.replace("'ve", " have")
        string = string.replace("n't", " not")
        string = string.replace("'re", " are")
        string = string.replace("'d", " would")
        string = string.replace("'ll", " will")
        string = string.replace("\r", " ")
        string = string.replace("\n", " ")
        string = re.sub(r'\d+', "number", string)
        string = ''.join(x for x in string if x.isalnum() or x == " ")
        string = re.sub(r'\s{2,}', " ", string)
        string = string.strip().lower()

        return string

    def preprocess(self):
        """
        Preprocesses the train text data 
        """
        print("Preprocessing started")
        with open(config.train_text_file_path, 'w') as text_file, open(config.train_labels_file_path, 'w') as labels_file:
            with open(config.train_pos_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    line = self._clean_text(line)
                    if len(line) > 0:
                        text_file.write(line + "\n")
                        labels_file.write("pos" + "\n")
            with open(config.train_neg_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    line = self._clean_text(line)
                    if len(line) > 0:
                        text_file.write(line + "\n")
                        labels_file.write("pos" + "\n")
        print("Processing complete ")




class ModelConfig:
    """
    Model configuration
    """

    def __init__(self):
        # vocab size after including special tokens
        self.vocab_size = 9203
        self.epochs = 20
        # batch setting
        self.batch_size = 128
        # layer sizes
        self.embedding_size = 300
        self.hidden_dim = 256
        self.style_hidden_dim = 8
        self.content_hidden_dim = 128
        # generative embedding dim = style_hidden_dim + content_hidden_dim
        self.generative_emb_dim = 136
        self.num_style = 2
        self.content_bow_dim = 7526
        # dropout
        self.dropout = 0.8
        # sequence length settings
        self.max_seq_len = 15
        # learning rates
        self.autoencoder_lr = 0.001
        self.style_adversary_lr = 0.001
        self.content_adversary_lr = 0.001
        # loss weights
        self.style_multitask_loss_weight = 10
        self.content_multitask_loss_weight = 3
        self.style_adversary_loss_weight = 1
        self.content_adversary_loss_weight = 0.03
        self.style_kl_lambda = 0.03
        self.content_kl_lambda = 0.03
        # kl annealing max iterations
        self.kl_anneal_iterations = 20000
        # noise
        self.epsilon = 1e-8
        self.label_smoothing = 0.1


if __name__ == "__main__":

    config = GeneralConfig()
    trythis = Preprocessor()
    trythis.preprocess()
