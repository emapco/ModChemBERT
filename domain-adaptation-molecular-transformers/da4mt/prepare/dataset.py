import argparse
import contextlib
import logging
import pathlib
import sys
import warnings

import pandas as pd

from da4mt.cli import add_prepare_dataset_args
from da4mt.utils import extract_physicochemical_props


def get_logger():
    logger = logging.getLogger("eamt.perpare.dataset")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


def validate_arguments(args):
    if not args.file.exists():
        raise ValueError(f"{args.file} does not seem to be a valid path")

    if args.file.suffix != ".csv":
        warnings.warn(f"Expecting a CSV file, got '{args.file.suffix}' instead.", stacklevel=2)

    if not args.output_dir.exists() or not args.output_dir.is_dir():
        raise ValueError(f"{args.output_dir} is not a directory.")


def make_descriptors(df: pd.DataFrame, output_dir: pathlib.Path, name: str):
    """Calculates and stores physicochemical properties

    :param df: Full dataframe containing SMILES and targets
    :param output_dir: Directory where the file will be saved
    :param name: Name of the dataset
    """
    logger = get_logger()

    extract_physicochemical_props(
        df["smiles"].tolist(),
        output_path=output_dir / f"{name}_mtr.jsonl",
        logger=logger,
        normalization_path=output_dir / f"{name}_normalization_values.json",
        subset="all",
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_prepare_dataset_args(parser)

    args = parser.parse_args()
    validate_arguments(args)
    return args


def make_data(args):
    validate_arguments(args)
    logger = get_logger()

    filename = args.file.stem

    logger.info(f"Loading {args.file} as CSV.")
    df = pd.read_csv(args.file)

    if df.columns[0].startswith("Unnamed:"):
        # Set that column as the index and convert index values to appropriate types
        df = df.set_index(df.columns[0])
        # Keep as is if conversion fails
        with contextlib.suppress(ValueError, TypeError):
            df.index = pd.to_numeric(df.index)

    if "smiles" not in df.columns:
        raise ValueError("CSV file must contain 'smiles' column.")

    # Make sure the dataset has a continous index for splitting
    df = df.reset_index(drop=True)
    # Some datasets in chembench have an extra explicit index column, we don't make us of it
    if "index" in df.columns:
        warnings.warn(
            "Dataset contains 'index' column. Assuming this is not a target and can be safely removed. "
            "Make sure this is intentional",
            stacklevel=2,
        )
        df = df.drop(columns=["index"])

    # Have everything in one folder even if redundant
    df.to_csv(args.output_dir / f"{filename}.csv", index=False, encoding="utf-8-sig")

    # Create features and training data once
    make_descriptors(df, args.output_dir, filename)


if __name__ == "__main__":
    make_data(get_args())
