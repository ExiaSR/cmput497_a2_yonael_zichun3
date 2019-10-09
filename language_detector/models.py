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
        self.n = 9

    def train(self):
        tokenizer = CharTokenizer()
        self.char_tokens = tokenizer.tokenize(self.text)
        char_grams = nltk.ngrams(self.char_tokens, self.n)
        self.len = len(self.char_tokens)
        self.vocabs = Vocabulary(self.char_tokens)
        self.dist = nltk.FreqDist(char_grams)

        if self.n > 1:
            self.char_counter = Counter(nltk.ngrams(self.char_tokens, self.n - 1))
        else:
            self.char_counter = Counter(self.char_tokens)

    def ngram_probaility(self, text_seq: tuple):
        char_occurence = self.char_counter[text_seq[:-1]]
        if not char_occurence:
            return 0.0
        # Natural log of P(Wn | Wn-1, n-N+1)
        log_prob = np.log(self.dist.freq(text_seq) / (char_occurence / self.len))
        logger.debug(text_seq)
        logger.debug("ln[P('{}'|{}) = {}]".format(text_seq[-1], text_seq[:-1], log_prob))
        return 0.0 if log_prob == float("-inf") else log_prob

    def perplexity(self, text):
        tokenizer = CharTokenizer()
        char_tokens = tokenizer.tokenize(text)
        char_grams = nltk.ngrams(char_tokens, self.n)
        log_prob = 0
        for token in char_grams:
            log_prob += self.ngram_probaility(token)

        return log_prob


class LaplaceModel(Model):
    def __init__(self, name, text):
        super().__init__(name, text)
        self.n = 2

    # TODO: Make it using characters not Bigrams
    def train(self):
        # https://www.nltk.org/_modules/nltk/probability.html
        tokenizer = CharTokenizer()
        self.char_tokens = tokenizer.tokenize(self.text)
        sentence = sent_tokenize(self.text)[0]
        vocabulary = set(self.text.split())
        cfdist = nltk.ConditionalFreqDist()

        for c in self.char_tokens:
            condition = len(c)
            cfdist[condition][c] += 1

        cpd_laplace = nltk.ConditionalProbDist(cfdist, nltk.LaplaceProbDist, bins=len(vocabulary))
        print([cpd_laplace[a].prob(b) for (a, b) in nltk.bigrams(sentence)])
        return cpd_laplace
        # cfd = nltk.FreqDist(nltk.ngrams(corpus, 1))

        # for c in cfd:
        #     print(c)


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
