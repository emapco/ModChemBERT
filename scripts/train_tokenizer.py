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

"""
Train a BPE tokenizer (wrapped as a transformers.PreTrainedTokenizerFast) from a
datasets.Dataset. Supports single / pair columns and sets a TemplateProcessing
post-processor so the fast tokenizer will behave correctly for single- and pair-
inputs (adds [CLS], [SEP], etc). Designed for maintainability and testability.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from typing import cast

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def dataset_text_iterator(ds: Dataset, text_col: str, text_pair_col: str | None = None) -> Iterator[str]:
    for i in range(len(ds)):
        row = ds[i]
        if text := row.get(text_col):
            yield str(text)
        if text_pair_col and (pair := row.get(text_pair_col)):
            yield str(pair)


def train_bpe_tokenizer(
    iterator: Iterable[str],
    *,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    unk_token: str = "[UNK]",
    pad_token: str = "[PAD]",
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    mask_token: str = "[MASK]",
    length: int | None = None,
    from_vocab: str | None = None,
    from_merges: str | None = None,
) -> Tokenizer:
    """
    Train a tokenizers.Tokenizer with a BPE model.
    Keep the pipeline explicit so we can swap in different normalizers / pre-tokenizers later.
    """
    special_tokens = [unk_token, pad_token, cls_token, sep_token, mask_token]
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token, vocab=from_vocab, merges=from_merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # type: ignore[call-arg]
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore[call-arg]
    if from_vocab:
        return tokenizer  # Skip training if loading from existing vocab

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=length)
    return tokenizer


def configure_postprocessor_and_wrap(
    tokenizer: Tokenizer,
    *,
    cls_token: str,
    sep_token: str,
    pad_token: str,
    unk_token: str,
    mask_token: str,
    model_max_length: int = 512,
) -> PreTrainedTokenizerFast:
    """
    After training we must set a template post-processor using the token ids assigned
    to special tokens. Wrapping into PreTrainedTokenizerFast makes the tokenizer
    compatible with transformers Trainer / pipelines.
    """
    # Resolve ids for special tokens (trainer already added them)
    cls_id = tokenizer.token_to_id(cls_token)
    mask_id = tokenizer.token_to_id(mask_token)
    pad_id = tokenizer.token_to_id(pad_token)
    sep_id = tokenizer.token_to_id(sep_token)
    unk_id = tokenizer.token_to_id(unk_token)
    if any(id_ is None for id_ in (cls_id, mask_id, pad_id, sep_id, unk_id)):
        raise ValueError("Expected special tokens to exist in the trained tokenizer vocab.")

    # TemplateProcessing makes the fast tokenizer produce correct sequences for single/pair inputs.
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{cls_token} $A {sep_token}",
        pair=f"{cls_token} $A {sep_token} $B {sep_token}",
        special_tokens=[
            (unk_token, unk_id),
            (cls_token, cls_id),
            (sep_token, sep_id),
            (pad_token, pad_id),
            (mask_token, mask_id),
        ],
    )  # type: ignore[call-arg]

    # Build the transformers wrapper. Passing tokenizer_object lets the PreTrainedTokenizerFast
    # reuse the trained tokenizers.Tokenizer without re-loading from disk.
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        pad_token=pad_token,
        cls_token=cls_token,
        sep_token=sep_token,
        mask_token=mask_token,
        model_max_length=model_max_length,
    )
    return fast


def save_tokenizer(fast_tokenizer: PreTrainedTokenizerFast, out_dir: str):
    """Save tokenizer for later reuse in transformers pipelines / training."""
    os.makedirs(out_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(out_dir)
    logger.info("Saved tokenizer to %s", out_dir)


@hydra.main(config_path="../conf", config_name="tokenizer", version_base="1.2")
def main(cfg: DictConfig):
    ds = (
        load_dataset(cfg.dataset.name, cfg.dataset.subset, split=cfg.dataset.split)
        if cfg.dataset.subset
        else load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    )

    try:
        record_count = len(cast("Dataset", ds))
        logger.info("Loaded dataset %s split=%s with %d records", cfg.dataset.name, cfg.dataset.split, record_count)
    except (TypeError, AttributeError):
        logger.info("Loaded dataset %s split=%s", cfg.dataset.name, cfg.dataset.split)

    if not isinstance(ds, Dataset):
        raise ValueError(f"Expected Dataset, got {type(ds)}. Please check your dataset configuration.")

    text_iter = dataset_text_iterator(ds, cfg.dataset.text_column, cfg.dataset.text_pair_column)
    tokenizer = train_bpe_tokenizer(
        text_iter,
        vocab_size=cfg.tokenizer.vocab_size,
        min_frequency=cfg.tokenizer.min_frequency,
        unk_token=cfg.special_tokens.unk_token,
        pad_token=cfg.special_tokens.pad_token,
        cls_token=cfg.special_tokens.cls_token,
        sep_token=cfg.special_tokens.sep_token,
        mask_token=cfg.special_tokens.mask_token,
        length=len(ds),
        from_vocab=cfg.tokenizer.from_vocab,
        from_merges=cfg.tokenizer.from_merges,
    )

    fast = configure_postprocessor_and_wrap(
        tokenizer,
        cls_token=cfg.special_tokens.cls_token,
        sep_token=cfg.special_tokens.sep_token,
        pad_token=cfg.special_tokens.pad_token,
        unk_token=cfg.special_tokens.unk_token,
        mask_token=cfg.special_tokens.mask_token,
        model_max_length=cfg.tokenizer.model_max_length,
    )

    save_tokenizer(fast, cfg.output.dir)


if __name__ == "__main__":
    main()
