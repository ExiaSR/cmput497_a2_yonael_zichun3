import click
import sys


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("model", type=str, required=True, nargs=-1)
@click.option("--debug", type=bool, default=False, help="Enable debug mode.")
def main(model, debug):
    """MODEL: --unsmoothed|--laplace|--interpolation"""
    if model[0] not in ["--unsmoothed", "--laplace", "--interpolation"]:
        print("Unsupported language model.")
        print('Try "main.py --help" for help.')
        sys.exit(1)


if __name__ == "__main__":
    main()
