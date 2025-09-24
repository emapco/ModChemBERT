import json
import logging
import os
import pathlib
from typing import Any, Literal

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from da4mt.utils import PhysicoChemcialPropertyExtractor
from modchembert.models.configuration_modchembert import ModChemBertConfig
from modchembert.models.modeling_modchembert import ModChemBertForSequenceClassification


def keep_subset(values, subset: Literal["all", "surface"], label_names: list[str], norm_std: list[float]):
    """
    Keep only the values for which the label is in the subset of labels and has non-zero standard deviation.
    :param values: List of arbitrary values
    :param subset: Subset of labels either 'all' or 'surface'.
    :param label_names: List of all labels in order.
    :param norm_std: List of standard deviations for each label.
    """
    if subset == "surface":
        keep = set(
            i
            for i, name in enumerate(label_names)
            if name in PhysicoChemcialPropertyExtractor.get_surface_descriptor_subset()
        )
    elif subset == "all":
        keep = range(len(label_names))
    else:
        raise ValueError(f"Invalid subset. Must be one of 'all', 'surface' not '{subset}.'")

    # Filter out indices with zero standard deviation
    return [v for i, v in enumerate(values) if i in keep and norm_std[i] != 0]


def add_id_labels_to_config(config: ModChemBertConfig, normalization_file: str, subset: Literal["all", "surface"]):
    """
    :param config: Preconfigured ModChemBertConfig
    :param normalization_file: Path to JSON file with mean, std and label_name for each target
    :param subset: Which subset of labels to include, either 'all' or 'surface'
    :return: Config including normalization/std/id2label and list of all labels. List of all labels includes also
        those not in the subset.
    """
    with open(normalization_file) as f:
        normalization_values = json.load(f)

    full_labels = normalization_values["label_names"]
    # Remove labels not in the subset
    config.id2label = {
        i: label
        for i, label in enumerate(
            keep_subset(normalization_values["label_names"], subset, full_labels, normalization_values["mean"])
        )
    }
    config.label2id = {v: k for k, v in config.id2label.items()}
    config.num_labels = len(config.id2label)
    config.problem_type = "regression"
    return config, full_labels


class preprocess_function:
    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, id2label, subset: Literal["all", "surface"], normalization_file: str
    ):
        self.tokenizer = tokenizer
        self.subset = subset
        self.label_names = id2label

        with open(normalization_file) as f:
            normalization_values = json.load(f)

        full_labels = normalization_values["label_names"]
        self.full_norm_std = normalization_values["std"]
        self.norm_mean = np.array(keep_subset(normalization_values["mean"], subset, full_labels, self.full_norm_std))
        self.norm_std = np.array(keep_subset(normalization_values["std"], subset, full_labels, self.full_norm_std))

    def __call__(self, examples):
        def _clean_property(x):
            return 0.0 if x == "" or "inf" in str(x) else float(x)

        batch_encoding = self.tokenizer(
            examples["smile"],
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        # Examples is a list of molecules, each with K properties
        labels = [
            keep_subset(
                [_clean_property(prop) for prop in labels],
                subset=self.subset,  # type: ignore
                label_names=self.label_names,
                norm_std=self.full_norm_std,
            )
            for labels in examples["labels"]
        ]
        if labels:
            labels_array = np.array(labels)
            # Ensure we only normalize up to the available norm_mean/norm_std length
            n_features = min(labels_array.shape[1], len(self.norm_mean), len(self.norm_std))
            # Apply normalization: (value - mean) / std
            if n_features > 0:
                std_slice = self.norm_std[:n_features]
                mean_slice = self.norm_mean[:n_features]
                labels_array[:, :n_features] = (labels_array[:, :n_features] - mean_slice) / std_slice
            labels = labels_array.tolist()

        batch_encoding["labels"] = labels
        return batch_encoding


def train_mtr(
    model: ModChemBertForSequenceClassification,
    tokenizer: PreTrainedTokenizerFast,
    training_args: TrainingArguments,
    dataset: DatasetDict,
    property_subset: Literal["all", "surface"],
    model_dir: str,
    logger: logging.Logger,
    orig_labels: list[str],
    normalization_file: str,
    save_model: bool = True,
):
    """
    :param model: (Pretrained) model instance
    :param tokenizer: tokenizer of this model
    :param training_args: HuggingFace Training arguments
    :param dataset: DatasetDict with train and validation split
    :param property_subset: Subset of the physico chemical properties to use
    :param model_dir: Output directory
    :param logger: Logger instance
    :param orig_labels: List with all labels
    :param normalization_file: Path to normalization file containing means and standard deviations
    :param save_model: If true saved the model and tokenizer in the model_dir
    :return:
    """
    logger.info(dataset)
    logger.info(dataset["train"])
    tokenized_dataset = dataset.map(
        preprocess_function(tokenizer, orig_labels, subset=property_subset, normalization_file=normalization_file),
        batched=True,
        remove_columns=["smile", "labels"],
    )

    # Some values are so large, that in float32 they become inf, we replace them by the maximum representable value
    def replace_inf_and_nan(samples):
        max_val = torch.finfo(samples["labels"].dtype).max
        min_val = torch.finfo(samples["labels"].dtype).min
        samples["labels"] = torch.nan_to_num(samples["labels"], 0.0, posinf=max_val, neginf=min_val)
        return samples

    tokenized_dataset.set_format(type="torch")
    tokenized_dataset = tokenized_dataset.map(replace_inf_and_nan, batched=True)

    logger.info(f"Training data has {tokenized_dataset['train'].num_rows:_} entries.")
    logger.info(tokenized_dataset["train"])

    # Use a padding collator so variable-length tokenized sequences are padded per-batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=data_collator,
    )
    trainer.train()

    final_output = training_args.output_dir or model_dir
    if save_model:
        logger.info(f"Saving model and tokenizer to {final_output}.")
        os.makedirs(final_output, exist_ok=True)
        model.save_pretrained(final_output)
        if tokenizer is not None:
            tokenizer.save_pretrained(final_output)

    if trainer.state.best_metric:
        assert not np.isnan(trainer.state.best_metric)
    return model, tokenizer, trainer.state.best_metric


def adapt_mtr(
    model_path: str,
    output_path: pathlib.Path,
    train_fp: pathlib.Path,
    normalization_fp: pathlib.Path,
    training_args: TrainingArguments,
    model_dict_config: dict[str, Any],
    logger: logging.Logger,
    save_model: bool = True,
    splits_fp: pathlib.Path | None = None,
) -> tuple[ModChemBertForSequenceClassification, PreTrainedTokenizerFast, float | None]:
    """
    Adapts the pretrained model using Multitask Regression

    :param model_path: Path to the directory containing pretrained model
    :param output_path: Path where the model will be saved
    :param train_fp: Path to the training file. Should be in jsonl format with keys being the molecules in SMILES and
        the values the phyisico chemical properties
    :param normalization_fp: Filepath to the normalization values for each property
    :param training_args: Training Arguments for HF Trainer
    :param model_dict_config: Configuration for the ModChemBert model
    :param logger: Logger
    :param save_model: Where model should be saved, useful for cross validation
    :param splits_fp: Optional path to file specifying splits. Should be a json file with keys "train", "test", "val"
        with indices for the respective splits as values.
    :return:
    """
    logger.info(f"Running domain adaptation with mtr on {model_path}.")

    ds = load_dataset("json", data_files=str(train_fp))

    if splits_fp is not None:
        with open(splits_fp) as f:
            split = json.load(f)
            ds = DatasetDict(
                train=ds["train"].select(split["train"]),  # type: ignore
                validation=ds["train"].select(split["val"]),  # type: ignore
            )
    logger.info(f"Normalization file: {normalization_fp}.")

    if training_args.bf16:
        dtype = torch.bfloat16
    elif training_args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model_config = ModChemBertConfig.from_pretrained(model_path, dtype=dtype)
    # Override with values from config - mainly to test classifier_pooling options
    model_config.update(model_dict_config)
    model_config, orig_labels = add_id_labels_to_config(model_config, str(normalization_fp), "all")
    property_subset = "all"
    assert model_config.num_labels != 0
    assert isinstance(model_config, ModChemBertConfig)

    model = ModChemBertForSequenceClassification.from_pretrained(model_path, config=model_config, dtype=dtype)
    model.register_for_auto_class("AutoModelForSequenceClassification")
    model_config.register_for_auto_class("AutoConfig")
    logger.info(model)

    return train_mtr(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        dataset=ds,  # type: ignore
        property_subset=property_subset,  # type: ignore
        model_dir=str(output_path),
        logger=logger,
        orig_labels=orig_labels,
        normalization_file=str(normalization_fp),
        save_model=save_model,
    )
