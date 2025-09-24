import argparse
import logging
import os
import pathlib
import sys

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import TrainingArguments

from da4mt.cli import add_adapt_args
from da4mt.training import adapt_mtr


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_adapt_args(parser)
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger("eamt.adapt")
    logger.setLevel(logging.DEBUG)

    print_handler = logging.StreamHandler(stream=sys.stderr)
    print_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    print_handler.setFormatter(formatter)

    logger.addHandler(print_handler)
    return logger


def load_training_args_and_model_cfg(
    output_path: pathlib.Path, dataset_name: str, hydra_config_name: str
) -> tuple[TrainingArguments, dict]:
    with initialize(config_path="../../../conf", version_base="1.2"):
        config = compose(config_name=hydra_config_name)

    training_args: TrainingArguments = instantiate(config.training_args)
    model_cfg = OmegaConf.to_container(config.modchembert_config, resolve=True)
    assert isinstance(model_cfg, dict), "modchembert_config could not be converted to dict"

    if training_args.output_dir is None:
        training_args.output_dir = str(output_path)
    else:
        training_args.output_dir = (output_path / (training_args.output_dir + "-" + dataset_name)).as_posix()
    if training_args.run_name is None:
        training_args.run_name = "-".join(str(output_path).split("/")[-2:])
    return training_args, model_cfg


def run_domain_adaptation(args):
    logger = get_logger()

    dataset_name = args.train_file.stem

    output_path: pathlib.Path = args.output
    training_args, model_cfg = load_training_args_and_model_cfg(
        output_path, f"{dataset_name.replace('_mtr', '')}", args.hydra_config_name
    )

    logger.info(f"Adapting model: {args.model}")
    os.environ["WANDB_RUN_GROUP"] = output_path.stem
    if args.normalization_file is None:
        raise ValueError("--normalization-file must be specified.")

    adapt_mtr(
        model_path=args.model,
        output_path=output_path,
        train_fp=args.train_file,
        normalization_fp=args.normalization_file,
        training_args=training_args,
        model_dict_config=model_cfg,
        logger=logger,
        save_model=True,
        splits_fp=args.splits_file,
    )


if __name__ == "__main__":
    run_domain_adaptation(get_args())
