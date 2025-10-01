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

import math
import os
from pathlib import Path
from typing import cast

import hydra
import optuna
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_utils import BestRun

from modchembert.dataset import ModChemBertDatasetProcessor
from modchembert.models.configuration_modchembert import ModChemBertConfig
from modchembert.models.modchembert import ModChemBert
from modchembert.types import TaskType

OmegaConf.register_new_resolver("truncate_path", lambda x: Path(str(x)).name.replace(".csv", ""))
OmegaConf.register_new_resolver(
    "truncate_path_only_parent", lambda x: f"{Path(str(x)).parent.name}".replace(".csv", "")
)


class ModChemBertTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_data_processor: ModChemBertDatasetProcessor | None = None
        self.eval_data_processor: ModChemBertDatasetProcessor | None = None
        self.mcb: ModChemBert | None = None
        self.training_args: TrainingArguments | None = None
        self._setup()

    def _setup(self):
        """Initialize model, datasets, and training arguments."""
        dtype = "float32"
        if self.cfg.training_args.get("bf16", None):
            dtype = "bfloat16"
        if self.cfg.training_args.get("fp16", None):
            dtype = "float16"

        mb_cfg: ModChemBertConfig = instantiate(self.cfg.modchembert_config)
        assert isinstance(mb_cfg, ModChemBertConfig)
        self.mcb = ModChemBert(
            **self.cfg.modchembert,
            dtype=dtype,
            config=mb_cfg,
        )

        seed = self.cfg.training_args.get("data_seed", self.cfg.training_args.get("seed", None))
        seed = int(seed) if seed is not None else None

        self.train_data_processor = ModChemBertDatasetProcessor(
            dataset_name=self.cfg.dataset.train.name,
            split=self.cfg.dataset.train.split,
            smiles_column=self.cfg.dataset.smiles_column,
            tokenizer=self.mcb.tokenizer,
            task=cast(TaskType, self.mcb.task),
            n_tasks=self.mcb.n_tasks,
            label_columns=self.cfg.dataset.label_columns,
            sample_size=(
                int(self.cfg.dataset.train.sample_size) if self.cfg.dataset.train.get("sample_size") else None
            ),
            deduplicate=self.cfg.dataset.train.get("deduplicate", False),
            is_task_finetune=self.cfg.dataset.get("is_task_finetune", False),
            is_benchmark=self.cfg.dataset.get("is_benchmark", False),
            apply_transforms=self.cfg.dataset.get("apply_transforms", False),
            seed=seed,
            num_proc=self.cfg.dataset.get("num_proc", None),
        )
        self.eval_data_processor = ModChemBertDatasetProcessor(
            dataset_name=self.cfg.dataset.eval.name,
            split=self.cfg.dataset.eval.split,
            smiles_column=self.cfg.dataset.smiles_column,
            tokenizer=self.mcb.tokenizer,
            task=cast(TaskType, self.mcb.task),
            n_tasks=self.mcb.n_tasks,
            label_columns=self.cfg.dataset.label_columns,
            sample_size=(int(self.cfg.dataset.eval.sample_size) if self.cfg.dataset.eval.get("sample_size") else None),
            deduplicate=self.cfg.dataset.train.get("deduplicate", False),
            is_task_finetune=self.cfg.dataset.get("is_task_finetune", False),
            is_benchmark=self.cfg.dataset.get("is_benchmark", False),
            apply_transforms=self.cfg.dataset.get("apply_transforms", False),
            seed=seed,
            num_proc=self.cfg.dataset.get("num_proc", None),
        )

        train_size = len(self.train_data_processor)
        eval_size = len(self.eval_data_processor)
        if train_size * int(self.cfg.training_args.num_train_epochs) == 0:
            raise ValueError(
                "Invalid training configuration: `train_size * num_train_epochs` must be greater than zero."
            )

        self.training_args = self._prepare_training_args(self.cfg, train_size)

        print(self.mcb.model)
        print(f"Model size: {self.mcb.model.num_parameters()} parameters.")
        print(f"Train size: {train_size}")
        print(f"Eval size: {eval_size}")

    def model_init_factory(self):
        """Factory function for model initialization during hyperparameter search."""

        def model_init(trial=None):
            assert self.mcb is not None, "ModChemBert model not initialized"

            if trial is None:
                return self.mcb.model

            # For hyperparameter optimization, create new model with trial parameters
            modern_bert_cfg = self.get_trial_modern_bert_cfg(trial, self.cfg)
            mb_cfg = ModChemBertConfig(**modern_bert_cfg)

            dtype = "float32"
            if self.cfg.training_args.get("bf16", None):
                dtype = "bfloat16"
            if self.cfg.training_args.get("fp16", None):
                dtype = "float16"

            mcb_trial = ModChemBert(
                **self.cfg.modchembert,
                dtype=dtype,
                config=mb_cfg,
            )
            return mcb_trial.model

        return model_init

    def _prepare_training_args(self, cfg: DictConfig, train_size: int) -> TrainingArguments:
        """Adjust training args from config: enums, seed, LR scaling, weight decay normalization."""
        ta = OmegaConf.to_container(cfg.training_args, resolve=True)
        assert isinstance(ta, dict)

        per_device_train_batch_size = int(ta.get("per_device_train_batch_size", 1))
        num_train_epochs = int(ta.get("num_train_epochs", 3))

        # Scale LR if enabled
        if cfg.get("scale_learning_rate") and cfg.scale_learning_rate.get("enabled", False):
            denom = float(cfg.scale_learning_rate.get("denominator", 128))
            base_lr = float(ta.get("learning_rate", 1e-4))
            scaled_lr = base_lr * math.sqrt(per_device_train_batch_size / denom)
            ta["learning_rate"] = scaled_lr
            print("--------------------------------------------------")
            print(f"Learning rate scaled to {ta.get('learning_rate', 0.0)}")
            print("--------------------------------------------------")

        # Normalized weight decay if enabled
        if bool(cfg.get("use_normalized_weight_decay", False)):
            if train_size * num_train_epochs <= 0:
                raise ValueError(
                    "Invalid training configuration: `train_size * num_train_epochs` "
                    "must be > 0 to normalize weight decay."
                )
            weight_decay = 0.05 * math.sqrt(per_device_train_batch_size / (train_size * num_train_epochs))
            ta["weight_decay"] = weight_decay
            print("--------------------------------------------------")
            print(f"Weight decay set to {ta.get('weight_decay', 0.0)}")
            print("--------------------------------------------------")

        return instantiate(OmegaConf.create(ta))

    @staticmethod
    def get_trial_modern_bert_cfg(trial: optuna.Trial | None, cfg: DictConfig):
        model_cfg = cfg.modchembert_config
        if trial is not None and hasattr(cfg.hyperopt.hp_space, "modchembert_config"):
            # Override config with trial parameters for model
            for param in cfg.hyperopt.hp_space.modchembert_config:
                param_name = param.name
                trial_name = f"modchembert_config__{param_name}"
                model_cfg[param_name] = trial.params[trial_name]

        return model_cfg

    @staticmethod
    def _get_trial_suggestion(trial: optuna.Trial, param_name: str, param: DictConfig):
        suggestion_map = {
            "int": lambda: trial.suggest_int(
                param_name,
                param.low,
                param.high,
                step=param.get("step", 1),
                log=param.get("log", False),
            ),
            "float": lambda: trial.suggest_float(
                param_name,
                param.low,
                param.high,
                step=param.get("step", None),
                log=param.get("log", False),
            ),
            "categorical": lambda: trial.suggest_categorical(param_name, param.choices),
        }
        return suggestion_map[param.type]()

    def _optuna_hp_space_factory(self):
        def optuna_hp_space_factory(trial: optuna.Trial):
            params = {}
            modern_bert_cfg = self.cfg.hyperopt.hp_space.get("modchembert_config", None)
            if modern_bert_cfg is not None:
                for param in modern_bert_cfg:
                    # prefixed and handle in model_init_factory
                    param_name = f"modchembert_config__{param.name}"
                    params[param_name] = self._get_trial_suggestion(trial, param_name, param)

            training_params = self.cfg.hyperopt.hp_space.get("training_args", None)
            if training_params is not None:
                for param in training_params:
                    param_name = f"{param.name}"  # no prefix: following default_hp_space_optuna example
                    params[param_name] = self._get_trial_suggestion(trial, param_name, param)

            return params

        return optuna_hp_space_factory

    def _init_trainer(self):
        """Initialize trainer for hyperparameter optimization."""
        assert self.mcb is not None, "ModChemBert model not initialized"
        assert self.train_data_processor is not None, "Training dataset not initialized"
        assert self.eval_data_processor is not None, "Evaluation dataset not initialized"
        training_args = self._prepare_training_args(self.cfg, len(self.train_data_processor))
        training_args.run_name = (
            None  # Remove run_name for hyperparameter optimization so wandb generates a unique name
        )

        callbacks = None
        early_stopping_patience = self.cfg.get("early_stopping_patience", None)
        if early_stopping_patience is not None:
            callbacks: list[TrainerCallback] | None = [
                (EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
            ]
        return Trainer(
            args=training_args,
            train_dataset=self.train_data_processor.dataset,
            eval_dataset=self.eval_data_processor.dataset,
            data_collator=self.mcb.data_collator,
            model_init=self.model_init_factory(),
            callbacks=callbacks,
        )

    def train(self):
        """Main training method that handles both regular training and
        hyperparameter optimization."""
        cfg = self.cfg

        # Ensure initialization is complete
        assert self.mcb is not None, "ModChemBert model not initialized"
        assert self.training_args is not None, "Training arguments not initialized"
        assert self.train_data_processor is not None, "Training dataset not initialized"
        assert self.eval_data_processor is not None, "Evaluation dataset not initialized"

        if not cfg.hyperopt.enabled:
            callbacks = None
            early_stopping_patience = self.cfg.get("early_stopping_patience", None)
            if early_stopping_patience is not None:
                callbacks: list[TrainerCallback] | None = [
                    (EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
                ]
            trainer = Trainer(
                model=self.mcb.model,
                args=self.training_args,
                data_collator=self.mcb.data_collator,
                train_dataset=self.train_data_processor.dataset,
                eval_dataset=self.eval_data_processor.dataset,
                callbacks=callbacks,
            )
            trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)

            final_dir = self.training_args.output_dir or ""
            os.makedirs(final_dir, exist_ok=True)
            self.mcb.model.save_pretrained(final_dir)
            if self.mcb.tokenizer is not None:
                self.mcb.tokenizer.save_pretrained(final_dir)
            return

        # Hyperparameter optimization
        trainer = self._init_trainer()

        storage = None
        if cfg.hyperopt.persistence:
            storage = optuna.storages.RDBStorage(
                url=cfg.hyperopt.storage_url,
                heartbeat_interval=cfg.hyperopt.storage_heartbeat_interval,
                engine_kwargs=cfg.hyperopt.storage_engine_kwargs,
            )

        best_trial = trainer.hyperparameter_search(
            n_trials=cfg.hyperopt.n_trials,
            gc_after_trial=True,
            direction="minimize",  # minimize loss
            backend="optuna",
            hp_space=self._optuna_hp_space_factory(),
            study_name=cfg.hyperopt.study_name,
            storage=storage,
            load_if_exists=cfg.hyperopt.load_if_exists,
            pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=cfg.hyperopt.patience),
        )
        self._save_best_trial_config(best_trial)

    def _save_best_trial_config(self, best_trial: BestRun | list[BestRun]):
        output_file = str(self.cfg.training_args.output_dir or "").removesuffix("/")
        best_trial_file = output_file + "-best_trial.json"
        with open(best_trial_file, "w") as f:
            import json

            if isinstance(best_trial, list):
                best_trial = best_trial[0]

            json.dump(best_trial._asdict(), f)


@hydra.main(config_path="../conf", config_name="mlm", version_base="1.2")
def main(cfg: DictConfig) -> None:
    trainer = ModChemBertTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
