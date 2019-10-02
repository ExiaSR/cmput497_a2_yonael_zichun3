import click
import csv
import sys
import os
import errno

from language_detector.models import Model
from typing import List


def get_training_files():
    (dirpath, _, filenames) = next(os.walk("data_train"))
    filenames = sorted(filenames)
    return [
        {"path": os.path.join(dirpath, filename), "name": filename}
        for filename in filenames
        if filename.endswith("txt.tra")
    ]


def get_dev_files():
    (dirpath, _, filenames) = next(os.walk("data_dev"))
    filenames = sorted(filenames)
    return [
        {"path": os.path.join(dirpath, filename), "name": filename}
        for filename in filenames
        if filename.endswith("txt.dev")
    ]


def train_model(model_type, name, text):
    model = Model.factory(model_type, name=name, text=text)
    model.train()

    return model


# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# https://stackoverflow.com/a/23794010
def safe_open_w(path, mode="wt"):
    mkdir_p(os.path.dirname(path))
    return open(path, mode)


def save_to_tsv(results, filename, output_dir="output"):
    """
    Archive language classification output into TSV file
    """
    with safe_open_w("{}/{}.tsv".format(output_dir, filename), "wt") as output_file:
        tsv_writer = csv.writer(output_file, delimiter="\t")
        rows = [
            [result["test_name"], result["model_name"], result["perplexity"], result["n"]]
            for result in results
        ]
        tsv_writer.writerows(rows)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("training_model_type", type=str, required=True, nargs=-1)
@click.option("--debug", type=bool, default=False, help="Enable debug mode.")
def main(training_model_type, debug):
    """TRAINING_MODEL_TYPE: --unsmoothed|--laplace|--interpolation"""
    if training_model_type[0] not in ["--unsmoothed", "--laplace", "--interpolation"]:
        print("Unsupported language model.")
        print('Try "main.py --help" for help.')
        sys.exit(1)

    training_model_type = training_model_type[0].replace("--", "")
    training_files = get_training_files()
    models: List(Model) = []

    # train model for each language
    for file in training_files:
        with open(file["path"], "r") as input_file:
            data = input_file.read()
            models.append(train_model(training_model_type, file["name"], data))

    dev_files = get_dev_files()
    dev_results = []
    for file in dev_files:
        # compute perplexity for each test file with each language models
        recorded_perplexity = []
        with open(file["path"], "r") as input_file:
            data = input_file.read()
            for model in models:
                recorded_perplexity.append(model.perplexity(data))
        best_model_idx = recorded_perplexity.index(min(recorded_perplexity))
        best_model = models[best_model_idx]
        dev_results.append(
            {
                "test_name": file["name"],
                "model_name": best_model.name,
                "perplexity": recorded_perplexity[best_model_idx],
                "n": best_model.n,
            }
        )
    save_to_tsv(dev_results, "results_dev_{}".format(training_model_type))


if __name__ == "__main__":
    main()
