# Copyright 2025 Emmanuel Cortes, All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import pathlib

import datasets
import torch
from transformers import PreTrainedTokenizerFast
from transformers.utils.generic import PaddingStrategy

from .types import VALID_TASK_TYPES, TaskType
from .util import log, normalization

dataset_transformations = {
    "delaney": normalization,
    "freesolv": normalization,
    "lipo": normalization,
    "clearance": log,
    "bace_regression": normalization,
}


class ModChemBertDatasetProcessor:
    """Dataset processing class for loading, preprocessing and tokenizing molecular property/language modeling datasets.

    This processor wraps common data-access patterns for ModChemBERT style experiments:
    1. Load a Hugging Face Hub dataset split OR a local CSV/JSON(L) file.
    2. Optionally resolve scaffold split indices for domain adaptation / task finetuning datasets.
    3. (Optionally) apply dataset-level numeric transformations (normalization / log) based on
       known benchmark dataset names.
    4. Filter to the SMILES column plus requested label columns (supports wildcard "*").
    5. Optional SMILES level deduplication.
    6. Optional random down-sampling with a reproducible seed.
    7. Batch tokenization into a Hugging Face `datasets.Dataset` ready for DataLoader usage.

    The resulting tokenized dataset is stored in `self.dataset`. For supervised tasks it will
    contain a `labels` tensor (shape: (batch_size,) for single-task or (batch_size, n_tasks) for multi-task).

    Parameters
    ----------
    dataset_name : str
        Path to a local data file (e.g. `data.csv`, `data.jsonl`) or a Hugging Face Hub dataset identifier.
        For task finetuning (domain adaptation) expects a companion `*_0.scaffold_splits.json` file
        alongside the CSV providing train/val/test indices.
    split : str
        Dataset split name when pulling from the Hub (e.g. "train", "validation", "test"). For
        scaffold-split datasets the value is matched fuzzily against "train", "val" (or "validation") and "test".
    smiles_column : str
        Name of the column containing canonical (or raw) SMILES strings to tokenize.
    tokenizer : transformers.PreTrainedTokenizerFast
        Tokenizer used to convert SMILES into model `input_ids` / `attention_mask` (and `special_tokens_mask` for MLM).
    task : TaskType, default="mlm"
        One of `VALID_TASK_TYPES` (e.g. "mlm", "regression", "classification", "mtr"). Determines label handling
        and padding strategy.
    n_tasks : int, default=1
        Number of prediction heads / target columns for supervised tasks (ignored for MLM).
    label_columns : list[str] | "*" | None, default=None
        Explicit list of label column names. Use "*" to take all non-SMILES columns. Ignored for MLM. If None,
        defaults to an empty list (unsupervised / no labels).
    sample_size : int | None, default=None
        If provided, randomly samples (after shuffle) up to this many rows from the filtered dataset.
    deduplicate : bool, default=False
        If True, keeps only the first occurrence of each unique SMILES string.
    is_task_finetune : bool, default=False
        If True, treat `dataset_name` as a local CSV requiring scaffold split JSON index file resolution.
    is_benchmark : bool, default=False
        If True, forces CSV loading without split selection (used for benchmark evaluation inputs).
    apply_transforms : bool, default=False
        If True, apply a numeric transformation (e.g. normalization or log) chosen by heuristic match
        against known dataset keys. Falls back to normalization if a match is found; leaves untouched otherwise.
    seed : int | None, default=None
        Random seed used for dataset shuffling before sampling. Ignored if `sample_size` is None.
    num_proc : int, optional, defaults to None
        Max number of processes when generating cache. Already cached shards are loaded sequentially.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        smiles_column: str,
        tokenizer: PreTrainedTokenizerFast,
        task: TaskType = "mlm",
        n_tasks: int = 1,
        label_columns: list[str] | None = None,
        sample_size: int | None = None,
        deduplicate: bool = False,
        is_task_finetune: bool = False,
        is_benchmark: bool = False,
        apply_transforms: bool = False,
        seed: int | None = None,
        num_proc: int | None = None,
    ) -> None:
        if task not in VALID_TASK_TYPES:
            raise ValueError(f"task must be one of {VALID_TASK_TYPES}, but got {task}")
        if label_columns is None:
            label_columns = []

        ds = self._load_dataset(dataset_name, split, is_task_finetune, is_benchmark)
        if smiles_column not in ds.column_names:
            raise KeyError(
                f"Column '{smiles_column}' not found in dataset split '{split}'. Available: {ds.column_names}"
            )

        # Apply transformations if requested
        ds = self._apply_transformations(dataset_name, ds, apply_transforms)

        # Handle wildcard case: use all columns except smiles_column
        if label_columns == "*":
            label_columns = [col for col in ds.column_names if col != smiles_column]

        # Keep only the smiles and labels columns to reduce memory footprint
        filtered_label_columns = [lc for lc in label_columns if lc in ds.column_names]
        if len(filtered_label_columns) != n_tasks and task != "mlm":
            raise ValueError(
                f"Expected {n_tasks} label columns, but found {len(filtered_label_columns)} in dataset. "
                f"Available: {ds.column_names}"
            )

        ds = ds.select_columns([smiles_column, *filtered_label_columns])
        ds = self._deduplicate(ds, smiles_column, deduplicate)
        ds = self._sample(ds, sample_size, seed)

        self.tokenizer = tokenizer
        self.smiles_column = smiles_column
        self.label_columns = filtered_label_columns
        self.task = task
        self.n_tasks = n_tasks
        self.dataset = ds.map(
            self._tokenize, batched=True, batch_size=2048, num_proc=num_proc, remove_columns=[self.smiles_column]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_dataset(
        self, dataset_name: str, split: str, is_task_finetune: bool, is_benchmark: bool
    ) -> datasets.arrow_dataset.Dataset:
        """Load a dataset from the Hugging Face Hub or local CSV file.
        For domain finetuning datasets (local CSV), expects a corresponding scaffold splits JSON file.
        """
        if is_benchmark:
            ds = datasets.load_dataset("csv", data_files=dataset_name, split="train")
            assert isinstance(ds, datasets.arrow_dataset.Dataset)
            return ds
        if not is_task_finetune:  # Pretraining
            # Check if a local file path was provided. If so, infer loader from extension.
            local_path = pathlib.Path(dataset_name)
            if local_path.exists() and local_path.is_file():
                ext = local_path.suffix.lower().lstrip(".")
                ext_map = {"jsonl": "json"}
                loader_name = ext_map.get(ext, ext)
                try:
                    print(f"Loading local dataset '{dataset_name}' with inferred loader '{loader_name}'...")
                    ds = datasets.load_dataset(loader_name, data_files=local_path.as_posix(), split=split)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load local dataset '{dataset_name}' with inferred loader '{loader_name}'."
                    ) from e
            else:
                # Remote or hub dataset identifier
                print(f"Loading dataset '{dataset_name}' from the Hugging Face Hub...")
                ds = datasets.load_dataset(dataset_name, split=split)
            assert isinstance(ds, datasets.arrow_dataset.Dataset)
            return ds

        suffix = pathlib.Path(dataset_name).suffix
        splits_filename = dataset_name.replace(suffix, "_0.scaffold_splits.json")
        try:
            with open(splits_filename) as f:
                split_indices = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find splits file '{splits_filename}' for dataset '{dataset_name}'. "
                "Expected for da4mt datasets."
            ) from e

        ds = datasets.load_dataset("csv", data_files=dataset_name, split="train")
        if "train" in split:
            ds = ds.select(split_indices["train"])  # type: ignore
        elif "test" in split:
            ds = ds.select(split_indices["test"])  # type: ignore
        else:
            ds = ds.select(split_indices["val"])  # type: ignore

        return ds

    def _deduplicate(self, ds: datasets.arrow_dataset.Dataset, smiles_column: str, deduplicate: bool):
        if not deduplicate:
            return ds

        print(f"Deduplicating dataset of size {len(ds)} by keeping first occurrence of each unique SMILES.")

        seen = set()

        def keep_first_occurrence(example):
            smiles_a = example[smiles_column]
            if smiles_a not in seen:
                seen.add(smiles_a)
                return True
            return False

        return ds.filter(keep_first_occurrence)

    def _sample(self, ds: datasets.arrow_dataset.Dataset, sample_size: int | None, seed: int | None = None):
        if sample_size is None:
            return ds

        print(f"Sampling {sample_size} entries from dataset of size {len(ds)}.")
        # Shuffle for variability, then take the first N
        ds = ds.shuffle(seed=seed)
        n = min(sample_size, len(ds))
        return ds.select(range(n))

    def _apply_transformations(
        self, dataset_name: str, ds: datasets.arrow_dataset.Dataset, apply_transforms: bool
    ) -> datasets.arrow_dataset.Dataset:
        """Apply deepchem transformations to the dataset if configured.

        This method checks if the dataset should have transformations applied based on
        the transformer_generators mapping and applies them if requested.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to check for transformations
        ds : datasets.arrow_dataset.Dataset
            The dataset to potentially transform
        apply_transforms : bool
            Whether to apply transformations

        Returns
        -------
        datasets.arrow_dataset.Dataset
            The transformed dataset or original dataset if no transformations applied
        """
        if not apply_transforms:
            return ds

        dataset_key = ""
        for key in dataset_transformations:
            if key in dataset_name.lower():
                dataset_key = key
                break

        transform_fn = dataset_transformations.get(dataset_key, normalization)
        print(f"Applying transformation '{transform_fn.__name__}' for dataset '{dataset_name}'.")
        return transform_fn(ds)

    def _tokenize(self, batch):
        smiles_batch = batch[self.smiles_column]
        if self.task == "mlm":
            tokens = self.tokenizer(
                smiles_batch,
                add_special_tokens=True,
                padding=True,
                pad_to_multiple_of=8,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            inputs = {
                "input_ids": tokens["input_ids"].squeeze(0),  # type: ignore
                "attention_mask": tokens["attention_mask"].squeeze(0),  # type: ignore
                "special_tokens_mask": tokens["special_tokens_mask"].squeeze(0),  # type: ignore
            }
            return inputs
        elif self.task in ["regression", "classification", "mtr"]:
            tokens = self.tokenizer(
                smiles_batch,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            labels = [batch[col] for col in self.label_columns]
            if len(labels) != 0:
                # Transpose the labels to get (batch_size, n_tasks)
                labels = list(zip(*labels, strict=True))
                if self.task == "regression" or self.task == "mtr":
                    label_dtype = torch.float32
                elif self.task == "classification":
                    label_dtype = torch.long if self.n_tasks == 1 else torch.float32
                else:
                    label_dtype = torch.float32

                labels = torch.tensor(labels, dtype=label_dtype)
                # For single task, squeeze the last dimension to get (batch_size,) shape
                if self.n_tasks == 1:
                    labels = labels.squeeze(-1)
            else:
                labels = None

            inputs = {**tokens, "labels": labels}
            return inputs
