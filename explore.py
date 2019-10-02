"""
Usage: python explore.py

Script to find the best `n` for each model, see output for detail.
"""
import re

from language_detector.models import Model
from language_detector.utils import *


def explore_model(model_type="unsmoothed"):
    training_files = get_training_files()
    dev_files = get_dev_files()

    n_list = range(1, 10)

    results = {}
    for n in n_list:
        models = [
            train_model(model_type, name=file["name"], text=file["data"], n=n)
            for file in training_files
        ]
        results[n] = [compute_lowest_perplexity(file, models) for file in dev_files]

    report = {}
    for n, result in results.items():
        num_of_mislabeld = 0
        for record in result:
            test_name = re.search(r"(.*)-(.*).txt.(tra|dev|test)", record["test_name"]).group(2)
            model_name = re.search(r"(.*)-(.*).txt.(tra|dev|test)", record["model_name"]).group(2)

            if test_name != model_name:
                num_of_mislabeld += 1

        report[n] = num_of_mislabeld
        print(
            "Model: {}, N: {}, Number of mislabeled test file: {}".format(
                model_type, n, num_of_mislabeld
            )
        )

    print("Best N for {} is {}".format(model_type, min(report, key=report.get)))


if __name__ == "__main__":
    model_types = ["unsmoothed"]
    for each in model_types:
        explore_model(each)
