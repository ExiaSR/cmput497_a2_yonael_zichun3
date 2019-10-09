import nltk
import logging
import numpy as np

from collections import Counter
from nltk.lm import NgramCounter
from nltk.lm import Vocabulary
from nltk.tokenize.simple import CharTokenizer
from nltk.probability import LaplaceProbDist
from nltk.tokenize import sent_tokenize

logger = logging.getLogger("cmput497")


class Model(object):
    def __init__(self, name, text):
        self.name = name
        self.text = text
        self.dist = None
        self.n = None
        self.text_grams = None
        self.char_tokens = None
        self.dist = None
        self.vocabs = None
        self.len = None

    def __str__(self):
        return "Model(name={}, n={})".format(self.name, self.n)

    def perplexity(self, text):
        pass

    @staticmethod
    def factory(type, **kwargs):
        if type == "unsmoothed":
            return UnsmoothedModel(**kwargs)
        elif type == "laplace":
            return LaplaceModel(**kwargs)
        elif type == "interpolation":
            return InterpolationModel(**kwargs)


class UnsmoothedModel(Model):
    def __init__(self, name, text):
        super().__init__(name, text)
        self.n = 1
        self.unk_threshold = 1

    def train(self):
        tokenizer = CharTokenizer()
        char_tokens = tokenizer.tokenize(self.text)
        vocabs = Vocabulary(char_tokens, unk_cutoff=self.unk_threshold)
        char_tokens = [token for token in char_tokens if token in vocabs]
        del vocabs # we dont need it anymore
        char_grams = nltk.ngrams(char_tokens, self.n)
        self.len = len(char_tokens)
        self.vocabs = Vocabulary(char_tokens)
        self.dist = nltk.FreqDist(char_grams)

        if self.n > 1:
            self.char_counter = Counter(nltk.ngrams(char_tokens, self.n - 1))
        else:
            self.char_counter = Counter(char_tokens)

    def ngram_probaility(self, text_seq: tuple):
        char_occurence = self.char_counter[text_seq[:-1]]
        numerator = self.dist.freq(text_seq)
        # for unigram, the numerator itself is already P(W), so just use 1 for denominator
        denominator = char_occurence / self.len if self.n > 1 else 1
        log_prob = np.log2(numerator / denominator)
        logger.debug("gram: {}, numerator: C{}={}, denominator: C{}={}".format(text_seq, text_seq, numerator, text_seq[:-1], denominator))
        logger.debug("ln[P('{}'|{}) = {}]".format(text_seq[-1], text_seq[:-1], log_prob))
        return log_prob

    def perplexity(self, text):
        tokenizer = CharTokenizer()
        # char_tokens = tokenizer.tokenize(text)
        char_tokens = [c if c in self.vocabs else "<UNK>" for c in tokenizer.tokenize(text) ]
        char_grams = nltk.ngrams(char_tokens, self.n)
        log_prob = 0
        for token in char_grams:
            log_prob += self.ngram_probaility(token)

        # 2 ^ (- 1/n * Sum(logp(w)))
        return np.power(2, - (1 / len(char_tokens) * log_prob))


class LaplaceModel(Model):
    def __init__(self, name, text):
        super().__init__(name, text)
        self.n = 4
       #self.unk_threshold = 1

    def train(self):
        tokenizer = CharTokenizer()
        char_tokens = tokenizer.tokenize(self.text)
        # vocabs = Vocabulary(char_tokens)
        # char_tokens = [token for token in char_tokens if token in vocabs]
        # del vocabs # we dont need it anymore
        char_grams = nltk.ngrams(char_tokens, self.n)
        self.len = len(char_tokens)
        self.vocabs = Vocabulary(char_tokens)
        self.dist = nltk.FreqDist(char_grams)

        if self.n > 1:
            self.char_counter = Counter(nltk.ngrams(char_tokens, self.n - 1))
        else:
            self.char_counter = Counter(char_tokens)

    def ngram_probaility(self, text_seq: tuple):
        char_occurence = self.char_counter[text_seq[:-1]]
        numerator = self.dist[text_seq] + 1
        # for unigram, the numerator itself is already P(W), so just use 1 for denominator
        denominator = char_occurence / self.len if self.n > 1 else self.len
        denominator += len(self.vocabs)
        log_prob = np.log2(numerator / denominator)
        logger.debug("gram: {}, numerator: C{}={}, denominator: C{}={}".format(text_seq, text_seq, numerator, text_seq[:-1], denominator))
        logger.debug("ln[P('{}'|{}) = {}]".format(text_seq[-1], text_seq[:-1], log_prob))
        return log_prob

    def perplexity(self, text):
        tokenizer = CharTokenizer()
        char_tokens = tokenizer.tokenize(text)
        # char_tokens = [c if c in self.vocabs else "<UNK>" for c in tokenizer.tokenize(text) ]
        char_grams = nltk.ngrams(char_tokens, self.n)
        log_prob = 0
        for token in char_grams:
            log_prob += self.ngram_probaility(token)

        # 2 ^ (- 1/n * Sum(logp(w)))
        return np.power(2, - (1 / len(char_tokens) * log_prob))

class InterpolationModel(Model):
    def __init__(self, name, text):
        super().__init__(name, text)
        self.n = 3

    def train(self):
        pass


def test():
    with open("data_train/udhr-eng.txt.tra", "r") as train_f, open(
        "data_dev/udhr-eng.txt.dev", "r"
    ) as dev_f:
        data = train_f.read().replace("\n", "")
        dev_data = dev_f.read().replace("\n", "")
        model = UnsmoothedModel("udhr-eng.txt.tra", data)
        model.train()
        print(model.perplexity(dev_data))


if __name__ == "__main__":
    test()
