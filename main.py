import click
import csv
import sys
import os
import errno
import logging

logger = logging.getLogger("cmput497")

from language_detector.models import Model
from language_detector.utils import (
    get_test_files,
    get_training_files,
    train_model,
    compute_lowest_perplexity,
)
# from language_detector.custom_types import Models


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
    with safe_open_w(os.path.join(output_dir, "{}.txt".format(filename)), "wt") as output_file:
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

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    """You must select one of the language models --unsmoothed|--laplace|--interpolation"""
    if not training_model_type:
        print("Unsupported language model.")
        print('Try "main.py --help" for help.')
        sys.exit(1)

    # train model for each language
    models = []
    training_files = get_training_files(train_dir)
    for file in training_files:
        models.append(train_model(training_model_type, file["name"], file["data"]))

    dev_files = get_test_files(test_dir)
    dev_results = []
    for file in dev_files:
        logger.debug("Testing: {}".format(file["name"]))
        dev_results.append(compute_lowest_perplexity(file, models))

    num_of_mislabeld = 0
    for record in dev_results:
        import re

        test_name = re.search(r"(.*)-(.*).txt.(tra|dev|tes)", record["test_name"]).group(2)
        model_name = re.search(r"(.*)-(.*).txt.(tra|dev|tes)", record["model_name"]).group(2)

        if test_name != model_name:
            num_of_mislabeld += 1

    logger.info("Number of mislabeled test file: {}".format(num_of_mislabeld))

    save_to_tsv(dev_results, "results_dev_{}".format(training_model_type if training_model_type != "laplace" else "add-one"), output_dir=out_dir)


if __name__ == "__main__":
    main()
