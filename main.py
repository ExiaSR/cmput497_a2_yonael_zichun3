import click
import csv
import sys
import os
import errno

from language_detector.models import Model
from language_detector.utils import (
    get_dev_files,
    get_training_files,
    train_model,
    compute_lowest_perplexity,
)
from language_detector.types import Models


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


@click.command()
@click.option(
    "--unsmoothed",
    "training_model_type",
    flag_value="unsmoothed",
    help="Use unsmoothed language model.",
)
@click.option("--laplace", "training_model_type", flag_value="laplace", help="Use one-hot model.")
@click.option(
    "--interpolation",
    "training_model_type",
    flag_value="interpolation",
    help="Use interpolation language model.",
)
@click.option("--train_dir", type=str, default="data_train", help="Path to training dataset.")
@click.option("--test_dir", type=str, default="data_dev", help="Path to test dataset.")
@click.option("--out_dir", type=str, default="output", help="Path to output tsv files.")
@click.option("--debug", type=bool, default=False, help="Enable debug mode.")
def main(training_model_type, train_dir, test_dir, out_dir, debug):
    """You must select one of the language models --unsmoothed|--laplace|--interpolation"""
    if not training_model_type:
        print("Unsupported language model.")
        print('Try "main.py --help" for help.')
        sys.exit(1)

    # train model for each language
    models: Models = []
    training_files = get_training_files(train_dir)
    for file in training_files:
        models.append(train_model(training_model_type, file["name"], file["data"]))

    dev_files = get_dev_files(test_dir)
    dev_results = []
    for file in dev_files:
        dev_results.append(compute_lowest_perplexity(file, models))

    save_to_tsv(dev_results, "results_dev_{}".format(training_model_type), output_dir=out_dir)


if __name__ == "__main__":
    main()
