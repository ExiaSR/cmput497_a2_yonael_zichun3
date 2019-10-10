"""
Usage: python langid.py

Produce result for both dev and test dataset.
"""
import os
import errno
import csv

from language_detector.utils import *


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
    with safe_open_w(os.path.join(output_dir, "{}.tsv".format(filename)), "wt") as output_file:
        tsv_writer = csv.writer(output_file, delimiter="\t")
        rows = [
            [result["test_name"], result["model_name"], result["perplexity"], result["n"]]
            for result in results
        ]
        tsv_writer.writerows(rows)


def explore_model(model_type="unsmoothed"):
    training_files = get_training_files()
    dev_files = get_test_files("data_dev")
    test_files = get_test_files("data_test")

    models = [
        train_model(model_type, name=file["name"], text=file["data"]) for file in training_files
    ]
    dev_results = [compute_lowest_perplexity(file, models) for file in dev_files]
    test_results = [compute_lowest_perplexity(file, models) for file in test_files]

    save_to_tsv(dev_results, "results_dev_{}".format(model_type))
    save_to_tsv(test_results, "results_test_{}".format(model_type))


if __name__ == "__main__":
    # TODO - add other language models to the list once they are done.
    model_types = ["unsmoothed", "laplace", "interpolation"]
    for each in model_types:
        explore_model(each)
