# ModChemBERT: ModernBERT as a Chemical Language Model

## Setup

Requirements:
- Python 3.11 # deepchem requires python<3.12

To install ModChemBERT and its dependencies, run:
```bash
make install
```
This creates a Python virtual environment in the `.venv` directory, installs the required Python packages, and clones the [`chemberta3`](https://github.com/emapco/chemberta3) and [`mergekit`](https://github.com/arcee-ai/mergekit) repositories.

Install flash attention for faster training and to take full advantage of ModernBERT's capabilities:
```bash
MAX_JOBS=4 make install-flash-attention
```
This installs the `flash-attn` package with parallel compilation using 4 CPU cores. Adjust `MAX_JOBS` as needed.
For details, refer to the [flash-attention repository](https://github.com/Dao-AILab/flash-attention).

## ModChemBERT Training Pipeline

```mermaid
flowchart TB
 subgraph C["Task-Adaptive Fine-Tuning"]
    direction TB
        C1["Task 1"]
        C2["Task 2"]
        Cdots["..."]
        CN["Task N"]
  end
    A["MLM Pretraining"] --> B["MTR of Physicochemical Properties (DAPT)"] & M["Checkpoint Averaging"]
    B --> C
    C1 --> M
    C2 --> M
    Cdots --> M
    CN --> M
    M --> G["Final ModChemBERT Model"]
```

### Datasets
### Datasets for Domain-Adaptive Pre-Training (DAPT) and Task-Adaptive Fine-Tuning (TAFT)

Sultan et al. (DA4MT) [1] utilized ADME and AstraZeneca datasets for DAPT of BERT-like models on multi-task regression (MTR) following MLM pretraining.

ModChemBERT uses the same datasets for DAPT and TAFT. They are split using DA4MT's [1] (Bemis-Murcko) scaffold split option (equivalent to DeepChem scaffold splitter). While DA4MT trains various models per dataset, ModChemBERT aggregates all datasets (`all_datasets_smiles`) during DAPT. For TAFT, the MTR model is fine-tuned separately on each dataset, producing 10 fine-tuned models.

#### Dataset Generation
Ensure Git LFS is installed to download the datasets:
```bash
sudo apt-get install git-lfs
git lfs install
```

Pull the datasets (CSV files) from the repository:
```bash
git lfs pull
```

To generate the datasets for MTR pretraining, run:
```bash
make prepare-domain-adaptation-data
```

This creates dataset files in `domain-adaptation-molecular-transformers/data`:
- `{dataset_name}_0.scaffold_splits.json`: contains indexes for train/val/test splits
- `{dataset_name}_mtr.jsonl`: contains physicochemical properties for MTR pretraining
- `{dataset_name}_normalization_values.json`: contains normalization values (mean and std) for each property
- `{dataset_name}.csv`: contains raw data for each dataset used for TAFT

### Datasets for ChemBERTa-3 Benchmarking

ModChemBERT utilizes the ChemBERTa-3 [2] benchmarking framework and datasets to evaluate downstream classification/regression tasks.
Additional classification datasets (antimalarial [3], cocrystal [4], and COVID-19 [5]) from Mswahili, et al. [6] are included to provide a more comprehensive evaluation.
The DA4MT regression datasets (ADME [7] and AstraZeneca) are also included in the benchmarking evaluation.
All benchmark datasets use DeepChem scaffold splits for train/val/test partitioning.

#### Dataset Generation
First, clone the ChemBERTa-3 repository:
```bash
make clone-chemberta3
```

Then generate the datasets for ChemBERTa-3 benchmarking:
```bash
make prepare-benchmarking-data
```

This creates task datasets in:
- `chemberta3/chemberta3_benchmarking/data/datasets/deepchem_splits`: Human-readable dataset splits for each task
- `chemberta3/chemberta3_benchmarking/data/featurized_datasets/deepchem_splits`: Dataset splits utilized by ChemBERTa-3 benchmarking framework

### Tokenizer
To train a BPE tokenizer on SMILES strings, run:
```bash
make train-tokenizer
```
Refer to `conf/tokenizer.yaml` for tokenizer training parameters.

You can build a tokenizer from an existing vocabulary by specifying the `from_vocab` and `from_merges` parameters.

### Pretraining
To pretrain ModChemBERT, run:
```bash
make pretrain
```

This performs MLM pretraining using `conf/mlm.yaml`; model parameters are specified in `conf/modchembert-config-mlm.yaml`.

If using a pretrained tokenizer, set its path in `conf/modchembert-config-mlm.yaml` under `modchembert.tokenizer_path`.

#### Hyperparameter Optimization
To perform hyperparameter optimization for MLM pretraining, run:
```bash
make hyperopt
```

Hyperparameter optimization is performed using Optuna and the search space is defined in `conf/hyperopt-mlm.yaml`.

ModChemBERT config hyperparameters to optimize are defined by `hyperopt.hp_space.modchembert_config`. `transformers.TrainingArguments` hyperparameters are set via `hyperopt.hp_space.training_args`.

### Domain-Adaptive Pre-Training (DAPT)
To perform DAPT using multi-task regression (MTR) on physicochemical properties, run:
```bash
make domain-adaptation-mtr MODEL_DIR=training_output/mlm/model_directory
```
Where `MODEL_DIR` is the path to the pretrained ModChemBERT model.

This performs MTR pretraining using `conf/domain-adaptation-mtr.yaml`; model parameters are specified in `conf/modchembert-config.yaml`.

### Task-Adaptive Fine-Tuning (TAFT)
To perform TAFT on each DA4MT task dataset, run:
```bash
python scripts/run_taft_roundrobin.py --num-workers 2 --cuda --pretrained-model-path training_output/model_directory 
```

The `--pretrained-model-path` argument is optional and should point to an MLM or DAPT checkpoint.
You can also set it in `conf/modchembert-config.yaml` under `modchembert.pretrained_model`.

TAFT produces multiple fine-tuned models. Training parameters are set in `conf/task-adaptation-ft.yaml`; model parameters are specified in `conf/modchembert-config.yaml`.

#### Hyperparameter Optimization
To perform hyperparameter optimization for TAFT, enable the hyperopt flag:
```bash
python scripts/run_taft_roundrobin.py --hyperopt --num-workers 2 --cuda --pretrained-model-path training_output/model_directory
```

Hyperparameter optimization is performed using Optuna and the search space is defined in `conf/hyperopt-taft.yaml`.

### Checkpoint Averaging
ModernBERT [8], JaColBERTv2.5 [9], and Llama 3.1 [10] demonstrate that checkpoint averaging (model merging) can yield a more performant final model. JaColBERTv2.5 [9] specifically notes gains in generalization without degrading out-of-domain performance.

ModChemBERT applies checkpoint averaging to integrate learned features from each task-adapted checkpoint and improve generalization.

To perform checkpoint averaging, run:
```bash
mergekit-yaml conf/merge/merge-taft-checkpoints.yaml training_output/model/path --trust-remote-code --copy-tokenizer [--cuda]
cp modchembert/models/modeling_modchembert.py training_output/model/path/
```

This merges TAFT checkpoints using `conf/merge/merge-taft-checkpoints.yaml`. The final merged model is saved to `training_output/model/path`. Add `--cuda` to use GPU.

Copying `modeling_modchembert.py` ensures the model can be loaded with `transformers.AutoModel.from_pretrained()`.

## ChemBERTa-3 Benchmarking
Evaluate ModChemBERT on downstream tasks using the ChemBERTa-3 benchmarking framework.

To evaluate on classification tasks, run:
```bash
make eval-classification
```

Configure classification datasets (antimalarial, bace_classification, etc.), metrics (roc_auc_score, prc_auc_score), and per-dataset hyperparameters in `conf/chemberta3/benchmark-classification.yaml`.

To evaluate on regression tasks, run:
```bash
make eval-regression
```

Configure regression datasets (esol, bace_regression, etc.), metrics (rms_score, mean_squared_error, mean_absolute_error, mae_score), transform options, and per-dataset hyperparameters in `conf/chemberta3/benchmark-regression.yaml`.

### Dataset-specific Hyperparameters
Each dataset has individual training hyperparameters:
- `batch_size`: Training batch size
- `epochs`: Number of training epochs  
- `learning_rate`: Learning rate for fine-tuning
- `classifier_pooling`: Pooling method for the classifier head
- `classifier_pooling_last_k`: Number of last hidden layers to use for pooling
- `classifier_pooling_attention_dropout`: Dropout rate for pooling attention (if applicable)
- `classifier_dropout`: Dropout rate for the classifier head
- `embedding_dropout`: Dropout rate for the embedding layer

### Benchmarking Outputs
Benchmarking logs are saved to `outputs`; model checkpoints go to `training_output/chemberta3-modchembert-ft-models`.

### Hyperparameter Optimization
To perform hyperparameter optimization on a ChemBERTa-3 benchmarking dataset, run:
```bash
.venv/bin/python scripts/train_modchembert.py \
  --config-dir=conf/benchmark-datasets \
  modchembert.pretrained_model=training_output/path/to/model \
  --config-name={dataset_config}
```

Where `{dataset_config}` is the name of a dataset config in `conf/benchmark-datasets` (e.g. `bace_classification.yaml`).

### Analysis
To analyze benchmarking results:
```bash
./scripts/analyze_log_directory.sh outputs/{benchmark_run_log_directory}
```
This script outputs a summary table of validation and test results per task plus overall averages.

## Classifier Pooling Methods
The ChemLM [11] paper explores pooling methods for chemical language models and finds the embedding method has the strongest impact on downstream performance among evaluated hyperparameters.

Behrendt et al. [12] noted that the last few layers contain task-specific information and that pooling methods leveraging information from multiple layers can enhance model performance. Their results further demonstrated that the `max_seq_mha` pooling method was particularly effective in low-data regimes, which is often the case for molecular property prediction tasks.

ModChemBERT further explores these pooling methods across DAPT, TAFT, and benchmarking phases.

Note: ModChemBERT’s `max_seq_mha` differs from MaxPoolBERT [12]. Behrendt et al. used PyTorch `nn.MultiheadAttention`, whereas ModChemBERT's `ModChemBertPoolingAttention` adapts ModernBERT’s `ModernBertAttention`. 
On ChemBERTa-3 benchmarks this variant produced stronger validation metrics and avoided the training instabilities (sporadic zero / NaN losses and gradient norms) seen with `nn.MultiheadAttention`. Training instability with ModernBERT has been reported in the past ([discussion 1](https://huggingface.co/answerdotai/ModernBERT-base/discussions/59) and [discussion 2](https://huggingface.co/answerdotai/ModernBERT-base/discussions/63)).

The available pooling methods are:
- `cls` - Last hidden layer [CLS] token (ModernBERT CLS)
- `mean` - Mean over last hidden layer (ModernBERT Mean)
- `max_cls` - Max over last k [CLS] tokens (MaxPoolBERT MaxCLS)
- `cls_mha` - MHA with [CLS] query (ModernBERT MHA)
- `max_seq_mha` - MHA with max pooled sequence as KV and max pooled [CLS] as query (MaxPoolBERT MaxSeq + MHA)
- `sum_mean` - Sum layers → mean tokens (ChemLM Sum + Mean)
- `sum_sum` - Sum layers → sum tokens (ChemLM Sum + Sum)
- `mean_mean` - Mean layers → mean tokens (ChemLM Mean + Mean)
- `mean_sum` - Mean layers → sum tokens (ChemLM Mean + Sum)
- `max_seq_mean` - Max over last k layers → mean tokens (custom)

## Citation
If you use ModChemBERT in your research, please cite the following:
```bibtex
@software{cortes-2025-modchembert,
  author = {Emmanuel Cortes},
  title = {ModChemBERT: ModernBERT as a Chemical Language Model},
  year = {2025},
  publisher = {GitHub},
  howpublished = {GitHub repository},
  url = {https://github.com/emapco/ModChemBERT}
}
```

## References
1. Sultan, Afnan, et al. "Transformers for molecular property prediction: Domain adaptation efficiently improves performance." arXiv preprint arXiv:2503.03360 (2025).
2. Singh R, Barsainyan AA, Irfan R, Amorin CJ, He S, Davis T, et al. ChemBERTa-3: An Open Source Training Framework for Chemical Foundation Models. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-4glrl-v2 This content is a preprint and has not been peer-reviewed.
3. Mswahili, M.E.; Ndomba, G.E.; Jo, K.; Jeong, Y.-S. Graph Neural Network and BERT Model for Antimalarial Drug Predictions Using Plasmodium Potential Targets. Applied Sciences, 2024, 14(4), 1472. https://doi.org/10.3390/app14041472
4. Mswahili, M.E.; Lee, M.-J.; Martin, G.L.; Kim, J.; Kim, P.; Choi, G.J.; Jeong, Y.-S. Cocrystal Prediction Using Machine Learning Models and Descriptors. Applied Sciences, 2021, 11, 1323. https://doi.org/10.3390/app11031323
5. Harigua-Souiai, E.; Heinhane, M.M.; Abdelkrim, Y.Z.; Souiai, O.; Abdeljaoued-Tej, I.; Guizani, I. Deep Learning Algorithms Achieved Satisfactory Predictions When Trained on a Novel Collection of Anticoronavirus Molecules. Frontiers in Genetics, 2021, 12:744170. https://doi.org/10.3389/fgene.2021.744170
6. Mswahili, M.E., Hwang, J., Rajapakse, J.C. et al. Positional embeddings and zero-shot learning using BERT for molecular-property prediction. J Cheminform 17, 17 (2025). https://doi.org/10.1186/s13321-025-00959-9
7. Cheng Fang, Ye Wang, Richard Grater, Sudarshan Kapadnis, Cheryl Black, Patrick Trapa, and Simone Sciabola. "Prospective Validation of Machine Learning Algorithms for Absorption, Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective" Journal of Chemical Information and Modeling 2023 63 (11), 3263-3274 https://doi.org/10.1021/acs.jcim.3c00160
8. Warner, Benjamin, et al. "Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference." arXiv preprint arXiv:2412.13663 (2024).
9. Clavié, Benjamin. "JaColBERTv2.5: Optimising Multi-Vector Retrievers to Create State-of-the-Art Japanese Retrievers with Constrained Resources." arXiv preprint arXiv:2407.20750 (2024).
10. Grattafiori, Aaron, et al. "The llama 3 herd of models." arXiv preprint arXiv:2407.21783 (2024).
11. Kallergis, G., Asgari, E., Empting, M. et al. Domain adaptable language modeling of chemical compounds identifies potent pathoblockers for Pseudomonas aeruginosa. Commun Chem 8, 114 (2025). https://doi.org/10.1038/s42004-025-01484-4
12. Behrendt, Maike, Stefan Sylvius Wagner, and Stefan Harmeling. "MaxPoolBERT: Enhancing BERT Classification via Layer-and Token-Wise Aggregation." arXiv preprint arXiv:2505.15696 (2025).
