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

TRAIN_ENV_BASE := PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 WANDB_MODE=disabled HF_HUB_OFFLINE=0
TRAIN_ENV_TOKENIZER := $(TRAIN_ENV_BASE) WANDB_PROJECT=ModChemBERT-Tokenizer
TRAIN_ENV_MODCHEMBERT := $(TRAIN_ENV_BASE) WANDB_PROJECT=ModChemBERT WANDB_LOG_MODEL=end
TASK_FINETUNE_ENV_MODCHEMBERT := $(TRAIN_ENV_BASE) WANDB_PROJECT=ModChemBERT-Task-Finetune-Hyperopt

# Installation targets
.PHONY: install
install: clone-chemberta3
	@if [ ! -d ".venv" ]; then \
		echo "Creating Python virtual environment..."; \
		python3 -m venv .venv; \
	fi
	pip install -e .[dev,adapt]
	make clone-mergekit

.PHONY: install-flash-attn
install-flash-attn:
	pip install flash-attn --no-build-isolation

.PHONY: clone-mergekit
clone-mergekit:
	@if [ ! -d "mergekit" ]; then \
		git clone https://github.com/arcee-ai/mergekit; \
	fi
	cd mergekit && pip install -e .

.PHONY: clone-chemberta3
clone-chemberta3:
	@if [ ! -d "chemberta3" ]; then \
		git clone https://github.com/emapco/chemberta3.git; \
	fi

# Pretraining targets
train-tokenizer:
	$(TRAIN_ENV_TOKENIZER) .venv/bin/python scripts/train_tokenizer.py

hyperopt:
	$(TRAIN_ENV_MODCHEMBERT) \
	.venv/bin/python scripts/train_modchembert.py hyperopt.enabled=true training_args.run_name=modchembert-hyperopt training_args.output_dir=training_output/modchembert-hyperopt

pretrain:
	$(TRAIN_ENV_MODCHEMBERT) \
	.venv/bin/python scripts/train_modchembert.py --config-name=mlm

pretrain-lg:
	$(TRAIN_ENV_MODCHEMBERT) \
	.venv/bin/python scripts/train_modchembert.py --config-name=mlm 'active_hyperparam_set=$${hyperparam_set_candidates.mid}' 'active_size=$${modern_bert_lg}'

pretrain-test:
	$(TRAIN_ENV_MODCHEMBERT) \
	.venv/bin/python scripts/train_modchembert.py --config-name=test

# Domain/task adaptation targets
prepare-domain-adaptation-data:
	cd domain-adaptation-molecular-transformers && bash prepare_data.sh

domain-adaptation-mtr:
	@if [ -z "$(MODEL_DIR)" ]; then \
		MODEL_DIR="training_output/release/ModChemBERT-MLM"; \
	else \
		MODEL_DIR="$(MODEL_DIR)"; \
	fi; \
	cd domain-adaptation-molecular-transformers && $(TRAIN_ENV_MODCHEMBERT) ../.venv/bin/python -m da4mt adapt \
	--model "../$$MODEL_DIR" \
	--train-file data/all_datasets_smiles_mtr.jsonl \
	--output ../training_output/dapt-mtr \
	--normalization-file data/all_datasets_smiles_normalization_values.json \
	--splits-file data/all_datasets_smiles_0.scaffold_splits.json \
	--hydra-config-name domain-adaptation-mtr

TASK_FINETUNE_COMMAND=$(TASK_FINETUNE_ENV_MODCHEMBERT) .venv/bin/python scripts/train_modchembert.py --config-dir=conf/task-adaptation-ft-datasets \
		modchembert.pretrained_model="training_output/release/ModChemBERT-MLM-DAPT" hyperopt.enabled=true
hyperopt-taft1:
	CUDA_VISIBLE_DEVICES=0 $(TASK_FINETUNE_COMMAND) --config-name=adme_ppb_h
	CUDA_VISIBLE_DEVICES=0 $(TASK_FINETUNE_COMMAND) --config-name=adme_microsom_stab_h
	CUDA_VISIBLE_DEVICES=0 $(TASK_FINETUNE_COMMAND) --config-name=adme_microsom_stab_r
hyperopt-taft2:
	CUDA_VISIBLE_DEVICES=0 $(TASK_FINETUNE_COMMAND) --config-name=adme_permeability
	CUDA_VISIBLE_DEVICES=0 $(TASK_FINETUNE_COMMAND) --config-name=adme_solubility
	CUDA_VISIBLE_DEVICES=0 $(TASK_FINETUNE_COMMAND) --config-name=astrazeneca_CL
hyperopt-taft3:
	CUDA_VISIBLE_DEVICES=1 $(TASK_FINETUNE_COMMAND) --config-name=astrazeneca_LogD74
hyperopt-taft4:
	CUDA_VISIBLE_DEVICES=1 $(TASK_FINETUNE_COMMAND) --config-name=adme_ppb_r
	CUDA_VISIBLE_DEVICES=1 $(TASK_FINETUNE_COMMAND) --config-name=astrazeneca_PPB
	CUDA_VISIBLE_DEVICES=1 $(TASK_FINETUNE_COMMAND) --config-name=astrazeneca_Solubility

taft-roundrobin:
	@if [ -z "$(MODEL_DIR)" ]; then \
		MODEL_DIR="training_output/release/ModChemBERT-MLM-DAPT"; \
	else \
		MODEL_DIR="$(MODEL_DIR)"; \
	fi; \
	.venv/bin/python scripts/run_taft_roundrobin.py --num-workers 2 --cuda --pretrained-model-path "$$MODEL_DIR"

# Merging targets
merge-taft-checkpoints:
	mergekit-yaml conf/merge/merge-taft-checkpoints.yaml training_output/merged/ModChemBERT-MLM-TAFT --cuda --trust-remote-code --copy-tokenizer
	cp modchembert/models/modeling_modchembert.py training_output/merged/ModChemBERT-MLM-TAFT/

merge-taft-checkpoints-dapt:
	mergekit-yaml conf/merge/merge-taft-checkpoints-dapt.yaml training_output/merged/ModChemBERT-MLM-DAPT-TAFT --cuda --trust-remote-code --copy-tokenizer
	cp modchembert/models/modeling_modchembert.py training_output/merged/ModChemBERT-MLM-DAPT-TAFT/

merge-taft-checkpoints-dapt-optimized:
	mergekit-yaml conf/merge/merge-taft-checkpoints-dapt-optimized.yaml training_output/merged/ModChemBERT-MLM-DAPT-TAFT-OPT --cuda --trust-remote-code --copy-tokenizer
	cp modchembert/models/modeling_modchembert.py training_output/merged/ModChemBERT-MLM-DAPT-TAFT-OPT/

# Benchmarking targets
prepare-benchmarking-data:
	cd chemberta3/chemberta3_benchmarking/data/data_preprocessing && $(TRAIN_ENV_MODCHEMBERT) bash prepare_data_script.sh

BENCHMARK_HYPEROPT_COMMAND=$(TRAIN_ENV_BASE) .venv/bin/python scripts/train_modchembert.py --config-dir=conf/benchmark-datasets \
		modchembert.pretrained_model=training_output/release/ModChemBERT-MLM-DAPT-TAFT-OPT
hyperopt-benchmark1:
	CUDA_VISIBLE_DEVICES=0 $(BENCHMARK_HYPEROPT_COMMAND) hyperopt.persistence=true hyperopt.n_trials=10 --config-name=hiv.yaml
hyperopt-benchmark2:
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=bace_classification.yaml
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=bace_regression.yaml
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=bbbp.yaml
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=clearance.yaml
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=clintox.yaml
hyperopt-benchmark3:
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=delaney.yaml
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=freesolv.yaml
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=lipo.yaml
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=sider.yaml
	CUDA_VISIBLE_DEVICES=1 $(BENCHMARK_HYPEROPT_COMMAND) --config-name=tox21.yaml

eval-classification:
	.venv/bin/python chemberta3/chemberta3_benchmarking/models_benchmarking/modchembert_benchmark/modchembert_finetune_classification.py

eval-regression:
	.venv/bin/python chemberta3/chemberta3_benchmarking/models_benchmarking/modchembert_benchmark/modchembert_finetune_regression.py

## Validation log analysis for comparison between models and hyperparameter sets
analyze-log-directory:
	@if [ -z "$(MODEL_DIR)" ]; then \
		echo "Usage: make analyze-log-directory MODEL_DIR=path/to/log/directory"; \
		echo "Example: make analyze-log-directory MODEL_DIR=chemberta3/chemberta3_benchmarking/models_benchmarking/modchembert_benchmark/logs_modchembert_regression_checkpoint-27088"; \
		exit 1; \
	fi
	@./scripts/analyze_log_directory.sh "$(MODEL_DIR)"

# Utility targets
lint:
	ruff check modchembert --fix --config pyproject.toml
	ruff format modchembert --config pyproject.toml
	ruff analyze graph --config pyproject.toml
	ruff clean

.PHONY: build
build:
	.venv/bin/python -m build --wheel --sdist

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
