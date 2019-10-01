from nltk import ngrams


class Model(object):
    def __init__(self, name, text):
        self.name = name
        self.text = text

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
        self.n = 3

    def train(self):
        pass


class LaplaceModel(Model):
    def __init__(self, name, text):
        super().__init__(name, text)
        self.n = 3

    def train(self):
        pass


class InterpolationModel(Model):
    def __init__(self, name, text):
        super().__init__(name, text)
        self.n = 3

    def train(self):
        pass
