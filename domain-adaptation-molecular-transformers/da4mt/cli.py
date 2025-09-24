import argparse
import pathlib


def add_prepare_dataset_args(parser: argparse.ArgumentParser):
    parser.add_argument("file", type=pathlib.Path, help="CSV files with smiles column and targets")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Directory where the preprocessed data will be saved",
    )

    parser.add_argument("--seed", type=int, default=0, help="Seed for stochastic data generation.")

    return parser


def add_prepare_splits_args(parser: argparse.ArgumentParser):
    parser.add_argument("file", type=pathlib.Path, help="CSV files with smiles column and targets")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Directory where the preprocessed data will be saved",
    )

    parser.add_argument(
        "--splitter",
        nargs="+",
        choices=["scaffold", "random"],
        required=True,
    )

    parser.add_argument("--num-splits", "-n", nargs="+", type=int, help="Number of k-fold splits")

    parser.add_argument("--seed", default=0, type=int, help="Seed for random splitting.")

    return parser


def add_prepare_parser(parser: argparse.ArgumentParser):
    from da4mt.prepare.dataset import make_data
    from da4mt.prepare.splits import make_splits

    subparsers = parser.add_subparsers(title="kind")
    dataset_parser = subparsers.add_parser("dataset", help="Run dataset preprocessing.")
    dataset_parser = add_prepare_dataset_args(dataset_parser)
    dataset_parser.set_defaults(func=make_data)

    splits_parser = subparsers.add_parser("splits", help="Split the dataset")
    splits_parser = add_prepare_splits_args(splits_parser)
    splits_parser.set_defaults(func=make_splits)

    return parser


def add_adapt_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", help="Pre-trained model to be adapted.", type=pathlib.Path)

    parser.add_argument("--train-file", help="Path to the training file", type=pathlib.Path)

    parser.add_argument("--output", help="Output directory of the adapted model", type=pathlib.Path)

    parser.add_argument(
        "--normalization-file",
        help="Path to the normalization values for MTR.",
        type=pathlib.Path,
    )

    parser.add_argument(
        "--splits-file",
        help="Path to a JSON file containing train/val/test splits.",
        type=pathlib.Path,
        default=None,
    )

    parser.add_argument(
        "--hydra-config-name",
        type=str,
        default="domain-adaptation-mtr",
        help="Name of the hydra config to use.",
    )

    return parser
