"""
Usage: python explore.py

Script to find the best `n` for each model, see output for detail.
"""
import re

from language_detector.models import Model
from language_detector.utils import *
from language_detector.custom_types import Models


def get_file_by_name(name, files):
    for file in files:
        if file["name"] == name:
            return file


def explore_model(model_type="unsmoothed"):
    training_files = get_training_files()
    dev_files = get_dev_files()

    n_list = range(1, 9)

    results = {}
    for n in n_list:
        models = [
            train_model(model_type, name=file["name"], text=file["data"], n=n)
            for file in training_files
        ]
        results[n] = [compute_lowest_perplexity(file, models) for file in dev_files]

    print("==========={}============".format(model_type))
    report = {}
    for n, result in results.items():
        num_of_mislabeld = 0
        for record in result:
            test_name = re.search(r"(.*)-(.*).txt.(tra|dev|test)", record["test_name"]).group(2)
            model_name = re.search(r"(.*)-(.*).txt.(tra|dev|test)", record["model_name"]).group(2)

            if test_name != model_name:
                num_of_mislabeld += 1

        report[n] = num_of_mislabeld
        print("N: {}, Accuracy: {:.6f}%".format(n, 100.0 * (1 - num_of_mislabeld / len(result))))

    best_n = min(report, key=report.get)
    print("\nn = {} for {} language model produced the best accuracy\n".format(best_n, model_type))

    # models = [
    #     train_model(model_type, name=file["name"], text=file["data"], n=best_n)
    #     for file in training_files
    # ]
    # result = [compute_lowest_perplexity(file, models) for file in dev_files]

    # error_tests = []
    # print("Error test cases:")
    # for record in result:
    #     test_name = re.search(r"(.*)-(.*).txt.(tra|dev|test)", record["test_name"]).group(2)
    #     model_name = re.search(r"(.*)-(.*).txt.(tra|dev|test)", record["model_name"]).group(2)

    #     if test_name != model_name:
    #         error_tests.append(
    #             {"test_name": record["test_name"], "model_name": record["model_name"]}
    #         )
    #         print("Test: {} Label: {}".format(record["test_name"], record["model_name"]))

    # print("\n")

    # for each in error_tests:
    #     dev_file = get_file_by_name(each["test_name"], dev_files)
    #     print("Test: {}".format(dev_file["name"]))
    #     results = sorted(compute_perplexity(dev_file, models), key=lambda i: i["perplexity"])
    #     for result in results:
    #         if result["perplexity"]:
    #             print(
    #                 "model: {}, perplexity: {}".format(result["model_name"], result["perplexity"])
    #             )
    #     print("\n")

    print("=================================\n")


if __name__ == "__main__":
    model_types = ["interpolation"]
    for each in model_types:
        explore_model(each)
