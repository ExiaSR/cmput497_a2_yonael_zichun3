import click
import sys
import os

from language_detector.models import Model


def get_training_files():
    (dirpath, _, filenames) = next(os.walk("data_train"))
    return [
        {"path": os.path.join(dirpath, filename), "name": filename}
        for filename in filenames
        if filename.endswith("txt.tra")
    ]


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("model", type=str, required=True, nargs=-1)
@click.option("--debug", type=bool, default=False, help="Enable debug mode.")
def main(model, debug):
    """MODEL: --unsmoothed|--laplace|--interpolation"""
    if model[0] not in ["--unsmoothed", "--laplace", "--interpolation"]:
        print("Unsupported language model.")
        print('Try "main.py --help" for help.')
        sys.exit(1)

    training_model_type = model[0].replace("--", "")
    training_files = get_training_files()
    models = []

    # train model for each language
    for file in training_files:
        with open(file["path"], "r") as input_file:
            data = input_file.read()
            training_model = Model.factory(training_model_type, name=file["name"], text=data)
            training_model.train()

            models.append(training_model)


if __name__ == "__main__":
    main()
