import sys
import os

from .models import Model
from .custom_types import Models

def get_training_files(dir="data_train"):
    if not os.path.isdir(dir):
        raise Exception("Directory \"{}\" does not exist.".format(dir))

    (dirpath, _, filenames) = next(os.walk(dir))
    filenames = sorted([filename for filename in filenames if filename.endswith("txt.tra")])
    files = []
    for filename in filenames:
        with open(os.path.join(dirpath, filename)) as input_f:
            data = input_f.read().replace("\n", " ")
            files.append({"path": os.path.join(dirpath, filename), "name": filename, "data": data})
    return files


def get_dev_files(dir="data_dev"):
    if not os.path.isdir(dir):
        raise Exception("Directory \"{}\" does not exist.".format(dir))

    (dirpath, _, filenames) = next(os.walk(dir))
    filenames = sorted([filename for filename in filenames if filename.endswith("txt.dev")])
    files = []
    for filename in filenames:
        with open(os.path.join(dirpath, filename)) as input_f:
            data = input_f.read().replace("\n", " ")
            files.append({"path": os.path.join(dirpath, filename), "name": filename, "data": data})
    return files


def get_test_files(dir="data_test"):
    if not os.path.isdir(dir):
        raise Exception("Directory \"{}\" does not exist.".format(dir))

    (dirpath, _, filenames) = next(os.walk(dir))
    filenames = sorted([filename for filename in filenames if filename.endswith("txt.test")])
    files = []
    for filename in filenames:
        with open(os.path.join(dirpath, filename)) as input_f:
            data = input_f.read().replace("\n", " ")
            files.append({"path": os.path.join(dirpath, filename), "name": filename, "data": data})
    return files

def train_model(model_type, name, text, n=None):
    model = Model.factory(model_type, name=name, text=text)
    if n:
        model.n = n
    model.train()

    return model

def compute_lowest_perplexity(test, models):
    recorded_perplexity = []
    for model in models:
        recorded_perplexity.append(model.perplexity(test["data"]))
    best_model_idx = recorded_perplexity.index(min(recorded_perplexity))
    best_model = models[best_model_idx]
    return {
        "test_name": test["name"],
        "model_name": best_model.name,
        "perplexity": recorded_perplexity[best_model_idx],
        "n": best_model.n,
    }

def compute_perplexity(test, models: Models):
    recorded_perplexity = []
    for model in models:
        recorded_perplexity.append(
            {
                "perplexity": model.perplexity(test["data"]),
                "model_name": model.name,
                "test_name": test["name"],
            }
        )

    return recorded_perplexity
