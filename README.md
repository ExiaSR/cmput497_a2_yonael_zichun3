# cmput497_a2_yonael_zichun3

In this assginment, we implemented varies n-grams language models, including unsmoothed, Laplace smoothing and deleted interpolation and use the models to compute the perplexity of testing dataset against the training corpus.

## Prerequisites

-   Python 3.7+
-   [virtualenv](https://virtualenv.pypa.io/en/latest/installation/)

## Setup

```sh
# Setup python virtual environment
$ virtualenv venv --python=python3
$ source venv/bin/activate

# Install python dependencies
$ pip install -r requirements.txt
```

## How to run

> `$ python main.py --unsmoothed`

> `$ python main.py --laplace --train_dir=custom_path_to_dataset --debug=True`

By default, the program assume the training dataset is located at `data_train`, and the test dataset is located at `data_dev`. See below for advanced usage.

Moreover, we assume all dataset files follow such naming convention, `(.*)-(.*).txt.(tra|dev|tes)`.

```
Usage: main.py [OPTIONS]

  You must select one of the language models
  --unsmoothed|--laplace|--interpolation

Options:
  --unsmoothed      Use unsmoothed language model.
  --laplace         Use one-hot model.
  --interpolation   Use interpolation language model.
  --train_dir TEXT  Path to training dataset.
  --test_dir TEXT   Path to test dataset.
  --debug BOOLEAN   Enable debug mode.
  --help            Show this message and exit.
```

## Authors

-   Yonael Bekele
-   Michael Lin
