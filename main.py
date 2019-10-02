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

    # train model for each language
    models: Models = []
    training_files = get_training_files()
    for file in training_files:
        models.append(train_model(training_model_type, file["name"], file["data"]))

    dev_files = get_dev_files()
    dev_results = []
    for file in dev_files:
        dev_results.append(compute_lowest_perplexity(file, models))

    save_to_tsv(dev_results, "results_dev_{}".format(training_model_type))


if __name__ == "__main__":
    main()
