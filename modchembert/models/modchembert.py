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
#
# Copyright 2017 PandeLab

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.typing import OneOrMany
from transformers import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..patch import patch_deep_chem_hf_model
from ..types import VALID_TASK_TYPES, TaskType
from .configuration_modchembert import ModChemBertConfig
from .modeling_modchembert import ModChemBertForMaskedLM, ModChemBertForSequenceClassification

# Model implementation adapted from DeepChem's ChemBERTa model
# deepchem/models/torch_models/chemberta.py  (v2.8.1-dev)
# Models may use float16/bfloat16 (e.g., for flash attention).
# This class override _predict() to ensure outputs are cast to float32 to support NumPy transforms.
patch_deep_chem_hf_model()  # patch load_from_pretrained and fit_generator


class ModChemBert(HuggingFaceModel):
    """ModChemBERT Model

    ModChemBERT is a transformer chemical language model for learning on SMILES strings.
    The model architecture is based on the ModernBERT architecture. The model
    has can be used for both pretraining an embedding and finetuning for
    downstream applications.

    The model supports two types of pretraining tasks - pretraining via masked language
    modeling and pretraining via multi-task regression. To pretrain via masked language
    modeling task, use task = `mlm` and for pretraining via multitask regression task,
    use task = `mtr`. The model supports the regression, classification and multitask
    regression finetuning tasks and they can be specified using `regression`, `classification`
    and `mtr` as arguments to the `task` keyword during model initialization.

    The model uses a tokenizer To create input tokens for the models from the SMILES strings.
    The default molformer tokenizer outperforms a trained BPE tokenizer (modchembert/bpe)
    trained on the PubChem 10M dataset and a smiles tokenizer (modchembert/smiles_tokenizer).

    Parameters
    ----------
    task: str
        The task defines the type of learning task in the model. The supported tasks are
         - `mlm` - masked language modeling commonly used in pretraining
         - `mtr` - multitask regression - a task used for both pretraining base models and finetuning
         - `regression` - use it for regression tasks, like property prediction
         - `classification` - use it for classification tasks
    config: ModChemBertConfig or dict, default None
        Configuration object for ModChemBert model or a dictionary containing configuration parameters. If None, default configuration is used.
    pretrained_model: str, default None
        Path to a pretrained ModChemBert model or the name of a pretrained model hosted on huggingFace. If None, the model is initialized with random weights.
    dtype: str or torch.dtype, default 'bfloat16'
        Data type for model parameters. Supported values are 'float32', 'float16', 'bfloat16' or the corresponding torch.dtype.
        Using 'float16' or 'bfloat16' enables faster training on GPUs.
    n_tasks: int, default 1
        Number of prediction targets for a multitask learning model
    tokenizer_path: str
        Path containing pretrained tokenizer used to tokenize SMILES string for model inputs. The tokenizer path can either be a huggingFace tokenizer model or a path in the local machine containing the tokenizer.
        If None, the default molformer tokenizer is used.
    data_collator_kwargs: dict, default None
        Dictionary of keyword arguments passed to the DataCollatorForLanguageModeling when task is `mlm`.
    **kwargs: dict
        Additional keyword arguments passed to the parent HuggingFaceModel class.

    Example
    -------
    >>> import os
    >>> import tempfile
    >>> tempdir = tempfile.mkdtemp()

    >>> # preparing dataset
    >>> import pandas as pd
    >>> import deepchem as dc
    >>> smiles = ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F","CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"]
    >>> labels = [3.112,2.432]
    >>> df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(["task1"], feature_field="smiles", featurizer=dc.feat.DummyFeaturizer())
    ...     dataset = loader.create_dataset(tmpfile.name)

    >>> # pretraining
    >>> from modchembert.modchembert import ModChemBert
    >>> pretrain_model_dir = os.path.join(tempdir, 'pretrain-model')
    >>> tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"
    >>> pretrain_model = ModChemBert(task='mlm', model_dir=pretrain_model_dir, tokenizer_path=tokenizer_path)  # mlm pretraining
    >>> pretraining_loss = pretrain_model.fit(dataset, nb_epoch=1)

    >>> # finetuning in regression mode
    >>> finetune_model_dir = os.path.join(tempdir, 'finetune-model')
    >>> finetune_model = ModChemBert(task='regression', model_dir=finetune_model_dir, tokenizer_path=tokenizer_path)
    >>> finetune_model.load_from_pretrained(pretrain_model_dir)
    >>> finetuning_loss = finetune_model.fit(dataset, nb_epoch=1)

    >>> # prediction and evaluation
    >>> result = finetune_model.predict(dataset)
    >>> eval_results = finetune_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.mae_score))
    """  # noqa: E501

    def __init__(
        self,
        task: TaskType,
        config: ModChemBertConfig | dict | None = None,
        pretrained_model: str | None = None,
        dtype: str | torch.dtype = "bfloat16",
        n_tasks: int = 1,
        tokenizer_path: str | None = None,
        data_collator_kwargs: dict[Any, Any] | None = None,
        **kwargs,
    ):
        if task not in VALID_TASK_TYPES:
            raise ValueError(f"task must be one of {VALID_TASK_TYPES}, but got {task}")
        if data_collator_kwargs is None:
            data_collator_kwargs = {}

        if pretrained_model is not None:
            if isinstance(config, dict):
                model_dict_config = config
            elif isinstance(config, ModChemBertConfig):
                model_dict_config = config.to_dict()
            else:
                model_dict_config = {}
            config = ModChemBertConfig.from_pretrained(pretrained_model)
            # Override with values from config - in case finetuning requires a different config
            config.update(model_dict_config)
        elif config is None:
            config = ModChemBertConfig()
        elif isinstance(config, dict):
            config = ModChemBertConfig.from_dict(config)
            config = config
        assert isinstance(config, ModChemBertConfig)

        if tokenizer_path is not None:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            tokenizer.model_max_length = config.max_position_embeddings
        elif pretrained_model is not None:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model)
        else:
            tokenizer_path = (Path(__file__).parent.parent / "tokenizers" / "molformer").as_posix()
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            tokenizer.model_max_length = config.max_position_embeddings

        if isinstance(dtype, str) and hasattr(torch, dtype):
            dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype), "dtype must be a torch.dtype or a string corresponding to a torch dtype"

        model: PreTrainedModel
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.sep_token_id
        config.bos_token_id = tokenizer.cls_token_id
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        config.dtype = dtype

        if task == "mlm":
            model = (
                ModChemBertForMaskedLM(config)
                if pretrained_model is None
                else ModChemBertForMaskedLM.from_pretrained(
                    pretrained_model,
                    config=config,
                    dtype=dtype,
                    ignore_mismatched_sizes=True,
                )
            )
            self.label_dtype = torch.long
            model.register_for_auto_class("AutoModelForMaskedLM")
        elif task == "mtr" or task == "regression":
            config.problem_type = "regression"
            config.num_labels = n_tasks
            model = (
                ModChemBertForSequenceClassification(config)
                if pretrained_model is None
                else ModChemBertForSequenceClassification.from_pretrained(
                    pretrained_model,
                    config=config,
                    dtype=dtype,
                    ignore_mismatched_sizes=True,
                )
            )
            self.label_dtype = dtype
            model.register_for_auto_class("AutoModelForSequenceClassification")
        elif task == "classification":
            if n_tasks == 1:
                config.problem_type = "single_label_classification"
                self.label_dtype = torch.long
            else:
                config.problem_type = "multi_label_classification"
                config.num_labels = n_tasks
                self.label_dtype = dtype
            model = (
                ModChemBertForSequenceClassification(config)
                if pretrained_model is None
                else ModChemBertForSequenceClassification.from_pretrained(
                    pretrained_model,
                    config=config,
                    dtype=dtype,
                    ignore_mismatched_sizes=True,
                )
            )
            model.register_for_auto_class("AutoModelForSequenceClassification")
        else:
            raise ValueError("invalid task specification")

        config.register_for_auto_class("AutoConfig")
        self.config = config
        self.n_tasks = n_tasks
        self.task = task
        self.tokenizer = tokenizer
        super().__init__(model=model, task=task, tokenizer=tokenizer, **kwargs)
        if self.task == "mlm":
            self.data_collator = DataCollatorForLanguageModeling(tokenizer, **data_collator_kwargs)

    def _prepare_batch(self, batch: tuple[Any, Any, Any]):
        """
        Prepares a batch of data for the model based on the specified task.
        It overrides the _prepare_batch of parent class for the following condition:-

        - When n_task == 1 and task == 'classification', CrossEntropyLoss is used
          which takes input in long int format.
        - When n_task > 1 and task == 'classification', BCEWithLogitsLoss is used
          which takes input in float format.
        """

        smiles_batch, y, w = batch

        if self.task == "mlm":
            tokens = self.tokenizer(
                smiles_batch[0].tolist(),
                add_special_tokens=True,
                padding=True,
                pad_to_multiple_of=8,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            inputs, labels = self.data_collator.torch_mask_tokens(tokens["input_ids"])
            inputs = {
                "input_ids": inputs.to(self.device),
                "labels": labels.to(self.device, dtype=self.label_dtype),
                "attention_mask": tokens["attention_mask"].to(self.device),
                "special_tokens_mask": tokens["special_tokens_mask"].to(self.device),
            }
            return inputs, None, w
        elif self.task in ["regression", "classification", "mtr"]:
            tokens = self.tokenizer(
                smiles_batch[0].tolist(),
                padding=True,
                pad_to_multiple_of=8,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            if y is not None:
                # y is None during predict
                y = torch.from_numpy(y[0])
                y = y.to(self.device, dtype=self.label_dtype)
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, "labels": y}
            return inputs, y, w

    def _predict(
        self,
        generator: Iterable[tuple[Any, Any, Any]],
        transformers: list[Transformer],
        uncertainty: bool,
        other_output_types: OneOrMany[str] | None = None,
    ):
        """Predicts output for data provided by generator.

        This is the private implementation of prediction. Do not
        call it directly. Instead call one of the public prediction methods.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        transformers: list of dc.trans.Transformers
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.
        uncertainty: bool
            specifies whether this is being called as part of estimating uncertainty.
            If True, it sets the training flag so that dropout will be enabled, and
            returns the values of the uncertainty outputs.
        other_output_types: list, optional
            Provides a list of other output_types (strings) to predict from model.

        Returns
        -------
            a NumPy array of the model produces a single output, or a list of arrays
            if it produces multiple outputs

        Note
        ----
        A HuggingFace model does not output uncertainty. The argument is here
        since it is also present in TorchModel. Similarly, other variables like
        other_output_types are also not used. Instead, a HuggingFace model outputs
        loss, logits, hidden state and attentions.
        """
        results: list[list[np.ndarray]] | None = None
        variances: list[list[np.ndarray]] | None = None
        if uncertainty and (other_output_types is not None):
            raise ValueError(
                "This model cannot compute uncertainties and other output types simultaneously. "
                "Please invoke one at a time."
            )
        if uncertainty:
            if self._variance_outputs is None or len(self._variance_outputs) == 0:
                raise ValueError("This model cannot compute uncertainties")
            if len(self._variance_outputs) != len(self._prediction_outputs):  # type: ignore
                raise ValueError("The number of variances must exactly match the number of outputs")
        if other_output_types and (self._other_outputs is None or len(self._other_outputs) == 0):
            raise ValueError("This model cannot compute other outputs since no other output_types were specified.")
        self._ensure_built()
        self.model.eval()
        for batch in generator:
            inputs, labels, weights = batch
            inputs, _, _ = self._prepare_batch((inputs, None, None))  # type: ignore

            # Invoke the model.
            output_values = self.model(**inputs)
            output_values = output_values.get("logits")

            # output_values may be float16 or bfloat16 - convert to float32 so that numpy can handle it
            output_values = output_values.to(dtype=torch.float32)

            if isinstance(output_values, torch.Tensor):
                output_values = [output_values]
            output_values = [t.detach().cpu().numpy() for t in output_values]

            # Apply transformers and record results.
            if uncertainty:
                var = [output_values[i] for i in self._variance_outputs]  # type: ignore
                if variances is None:
                    variances = [var]
                else:
                    for i, t in enumerate(var):
                        variances[i].append(t)
            access_values = []
            if other_output_types:
                access_values += self._other_outputs  # type: ignore
            elif self._prediction_outputs is not None:
                access_values += self._prediction_outputs

            if len(access_values) > 0:
                output_values = [output_values[i] for i in access_values]

            if len(transformers) > 0:
                if len(output_values) > 1:
                    raise ValueError("predict() does not support Transformers for models with multiple outputs.")
                elif len(output_values) == 1:
                    output_values = [undo_transforms(output_values[0], transformers)]
            if results is None:
                results = [[] for i in range(len(output_values))]
            for i, t in enumerate(output_values):
                results[i].append(t)

        # Concatenate arrays to create the final results.
        final_results = []
        final_variances = []
        if results is not None:
            for r in results:
                final_results.append(np.concatenate(r, axis=0))

        if uncertainty and variances is not None:
            for v in variances:
                final_variances.append(np.concatenate(v, axis=0))
            return zip(final_results, final_variances, strict=False)

        if len(final_results) == 1:
            return final_results[0]
        else:
            return np.array(final_results)
