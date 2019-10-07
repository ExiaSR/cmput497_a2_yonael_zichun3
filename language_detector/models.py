import nltk
import numpy as np

from nltk.lm import NgramCounter
from nltk.lm import Vocabulary
from nltk.probability import LaplaceProbDist
from nltk.tokenize.simple import CharTokenizer
from nltk.tokenize import sent_tokenize


class Model(object):
    def __init__(self, name, text):
        self.name = name
        self.text = text
        self.n = None
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
        self.n = 2

    def train(self):
        self.word_tokens = nltk.tokenize.word_tokenize(self.text)
        # https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity/33269399
        char_grams = nltk.ngrams(
            self.char_tokens,
            self.n,
            pad_right=True,
            pad_left=True,
            left_pad_symbol="<s>",
            right_pad_symbol="</s>",
        )
        self.word_tokens.extend(["<s>", "</s>"])
        self.len = len(self.word_tokens)
        self.vocabs = Vocabulary(self.word_tokens)
        self.dist = nltk.FreqDist(char_grams)

    def ngram_probaility(self, text_seq: tuple):
        log_prob = np.log(
            self.dist.freq(text_seq) / (self.word_tokens.count(text_seq[0]) / self.len)
        )
        # print("{} = {} / ({} / {})".format(log_prob, self.dist.freq(text_seq), self.word_tokens.count(text_seq[0]), self.len))
        # TODO
        # - How to handle zeor probaility? for now we return 0
        return 0.0 if log_prob == float("-inf") else log_prob

    def perplexity(self, text):
        word_tokens = nltk.word_tokenize(text)
        char_grams = nltk.ngrams(
            word_tokens,
            self.n,
            pad_right=True,
            pad_left=True,
            left_pad_symbol="<s>",
            right_pad_symbol="</s>",
        )
        log_prob = 0

        for token in char_grams:
            if token[0] in self.vocabs:
                log_prob += self.ngram_probaility(token)

        return log_prob


class LaplaceModel(Model):
    def __init__(self, name, text):
        super().__init__(name, text)
        self.n = 2

    # TODO: Make it using characters not Bigrams
    def train(self):
        # https://www.nltk.org/_modules/nltk/probability.html

        corpus = self.text.split()
        sentence = sent_tokenize(self.text)[0]
        vocabulary = set(corpus)
        cfd = nltk.ConditionalFreqDist(nltk.bigrams(corpus))
        
        cpd_laplace = nltk.ConditionalProbDist(cfd, nltk.LaplaceProbDist, bins=len(vocabulary))
        print([cpd_laplace[a].prob(b) for (a,b) in nltk.bigrams(sentence)])

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
        data = train_f.read()
        dev_data = dev_f.read()  # "liberty and the security of person" #dev_f.read()
        model = UnsmoothedModel("udhr-eng.txt.tra", data)
        model.train()
        print(model.perplexity(dev_data))


if __name__ == "__main__":
    test()

