# cmput497_a2_yonael_zichun3

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

```
Usage: main.py [OPTIONS] TRAINING_MODEL_TYPE...

  TRAINING_MODEL_TYPE: --unsmoothed|--laplace|--interpolation

Options:
  --debug BOOLEAN  Enable debug mode.
  --help           Show this message and exit.
```

## Authors

-   Yonael Bekele
-   Michael Lin
