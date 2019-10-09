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
        del vocabs  # we dont need it anymore
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
        logger.debug(
            "gram: {}, numerator: C{}={}, denominator: C{}={}".format(
                text_seq, text_seq, numerator, text_seq[:-1], denominator
            )
        )
        logger.debug("ln[P('{}'|{}) = {}]".format(text_seq[-1], text_seq[:-1], log_prob))
        return log_prob

    def perplexity(self, text):
        tokenizer = CharTokenizer()
        # char_tokens = tokenizer.tokenize(text)
        char_tokens = [c if c in self.vocabs else "<UNK>" for c in tokenizer.tokenize(text)]
        char_grams = nltk.ngrams(char_tokens, self.n)
        log_prob = 0
        for token in char_grams:
            log_prob += self.ngram_probaility(token)

        # 2 ^ (- 1/n * Sum(logp(w)))
        return np.power(2, -(1 / len(char_tokens) * log_prob))


class LaplaceModel(Model):
    def __init__(self, name, text):
        super().__init__(name, text)
        self.n = 3

    def train(self):
        tokenizer = CharTokenizer()
        char_tokens = tokenizer.tokenize(self.text)
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
        denominator = char_occurence / self.len if self.n > 1 else self.len
        denominator += len(self.vocabs)
        log_prob = np.log2(numerator / denominator)
        logger.debug(
            "gram: {}, numerator: C{}={}, denominator: C{}={}".format(
                text_seq, text_seq, numerator, text_seq[:-1], denominator
            )
        )
        logger.debug("ln[P('{}'|{}) = {}]".format(text_seq[-1], text_seq[:-1], log_prob))
        return log_prob

    def perplexity(self, text):
        tokenizer = CharTokenizer()
        char_tokens = tokenizer.tokenize(text)
        char_grams = nltk.ngrams(char_tokens, self.n)
        log_prob = 0
        for token in char_grams:
            log_prob += self.ngram_probaility(token)

        # 2 ^ (- 1/n * Sum(logp(w)))
        return np.power(2, -(1 / len(char_tokens) * log_prob))


class InterpolationModel(Model):
    def __init__(self, name, text):
        super().__init__(name, text)
        self.n = 1
        self.multi_grams = [None for i in range(self.n)]
        self.multi_grams_dist = [None for i in range(self.n)]
        self.weights = []

    def __str__(self):
        return "InterpolationModel(name={}, n={}, weights={})".format(
            self.name, self.n, self.weights
        )

    def train(self):
        tokenizer = CharTokenizer()
        char_tokens = tokenizer.tokenize(self.text)
        self.len = len(char_tokens)
        #print('this is the len {}'.format(self.len))
        print("File = {}".format(self.name))
        for n in range(1, self.n + 1):
            print("this is {}".format(n))
            print("this is the current len of multi_grams {}".format(len(self.multi_grams)))
            self.multi_grams[n - 1] = nltk.ngrams(char_tokens, n)
            self.multi_grams_dist[n - 1] = nltk.FreqDist(nltk.ngrams(char_tokens, n))
        self.deleted_interpolation()

    def deleted_interpolation(self):
        n = self.n
        weights = [0 for i in range(self.n)]
        for gram in self.multi_grams[n - 1]:
            if self.multi_grams_dist[n - 1][gram] > 0:
                tmp = [0 for i in range(self.n)]
                cnt = len(gram) - 2  # magic number to help with list slicing for denominator
                for i in range(1, n + 1):
                    numerator = self.multi_grams_dist[i - 1][gram[-i:]] - 1 # Count(t1, t2, t3) - 1
                    if i == 1:
                        # print('self.len before {}'.format(self.len))
                        denominator = self.len - 1  
                        # print('self.len after {}'.format(denominator))
                        
                    else:
                        denominator = self.multi_grams_dist[i - 2][gram[cnt:-1]] - 1
                
                    logger.debug(
                        "n: {}, gram: {}, numerator: C{}={}, denominator: {}={}".format(
                            i,
                            gram,
                            gram[-i:],
                            numerator,
                            "C{}".format(gram[cnt:-1]) if i > 1 else "N - 1",
                            denominator,
                        )
                    )
                    if i > 1:
                        cnt -= 1
                    tmp[i - 1] = float(numerator) / float(denominator) if denominator > 0 else 0.0
                idx = tmp.index(max(tmp)) # get the index of max "value"
                weights[idx] += self.multi_grams_dist[n - 1][gram] # increment lambda_idx by C(t_i, t_i-n)
        # print("{} {}".format(self.name, weights))
        self.weights = [np.divide(w, np.sum(weights)) for w in weights]

    def ngram_probaility(self, text_seq: tuple, n: int, cnt: int):
        try:
            weight = self.weights[n]
            numerator = self.multi_grams_dist[n][text_seq[-(n + 1) :]]
            denominator = self.len if n == 0 else self.multi_grams_dist[n - 2][text_seq[cnt:-1]]
            logger.debug(
                "n: {}, gram: {}, numerator: C{} = {}, denominator: C{} = {}".format(
                    n + 1, text_seq, text_seq[-(n + 1) :], numerator, text_seq[cnt:-1], denominator
                )
            )
            return weight * float(numerator) / float(denominator) if denominator > 0 else 0.0
        except Exception:
            return 0.0

    def perplexity(self, text):
        tokenizer = CharTokenizer()
        char_tokens = tokenizer.tokenize(text)
        char_grams = nltk.ngrams(char_tokens, self.n)
        log_prob = 0
        for token in char_grams:
            # calculate weighted probaility
            cnt = len(token) - 2
            prob = 0.0
            for n in range(self.n):
                prob += self.ngram_probaility(token, n, cnt)
                if n > 0:
                    cnt -= 1
            log_prob += np.log2(prob)
        return np.power(2, -(1 / len(char_tokens) * log_prob))

def test():
    with open("data_train/udhr-eng.txt.tra", "r") as train_f, open(
        "data_dev/udhr-kin.txt.dev", "r"
    ) as dev_f:
        data = train_f.read().replace("\n", "")
        dev_data = dev_f.read().replace("\n", "")
        model = InterpolationModel("udhr-eng.txt.tra", data)
        model.train()
        print(model.perplexity(dev_data))
        print(model)


if __name__ == "__main__":
    test()
