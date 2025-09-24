# Transformers for Molecular Property Prediction: Domain Adaptation Efficiently Improves Performance
Pretrained and domain adapted models appearing in our publication can now be found on [huggingface](https://huggingface.co/collections/UdS-LSV/domain-adaptation-molecular-transformers-6821e7189ada6b7d0a5b62d4). 
An example of domain adaptation using multi-task regression is available [here](https://github.com/uds-lsv/domain-adaptation-molecular-transformers/blob/publication/mtr-example/mtr-domain-adaptation.ipynb). All necessary step can be implemented roughly [200 lines of python code](https://github.com/uds-lsv/domain-adaptation-molecular-transformers/blob/publication/mtr-example/utils.py).

### Dependencies
```bash
pip install -r requirements.txt
```

We also need the `useful_rdkit_utils` package, which requires `python>=3.11` but still works fine with
our python version.
```
pip install --ignore-requires-python useful_rdkit_utils==0.74
```

### Overview
In general all steps in the pipeline are accessible as subcommands to the command line interface of the `da4mt` package.
```bash
$ python -m da4mt --help

usage: __main__.py [-h] {prepare,adapt} ...

options:
  -h, --help            show this help message and exit

Command:
  {prepare,adapt}
    prepare             Run data preparation.
    adapt               Run domain adaptation.
```

### Datasets & Preprocessing

#### Downstream datasets
In general the dataset preprocessing encompasses 2 steps:
1. Precomputing the necessary labels for pretraining and domain adaptation, i.e. Physiochemical properties and triples for contrastive learning
2. Splitting the datasets into `k` folds. `k` needs to be determined by hand

The first step can be done by running

```bash
python -m da4mt prepare dataset <csvfile> -o <outputdir>
```
The csvfile should contain a `smiles` column, all other columns will interpreted as labels.
This produces the following files in `<outputdir>`, where name is the basename of the `<csvfile>`, e.g. if `csvfile=/some/path/to/file.csv`, then `name=file`.
```
<name>.csv         <-- Copy of the full dataset
<name>_mtr.jsonl   <-- Physiochemcial properties for MTR
<name>_normalization_values.json  <-- Mean and std for each MTR label
```

To split the files, run `python -m da4mt prepare splits <csvfile> -o <outputdir> --splitter random scaffold --num-splits <k1> <k2> <k3>`. `k*` is the number of splits and needs to be determined by hand, i.e. not all datasets may be 5-fold scaffold splittable, in this case the program will exit with an error.

This will create `k` files for each splitter:
```
<name>_<i>.<splitter>_splits.json
```
The files contains the indices for the `train`, `val` and `test` splits as a dictionary.

### Domain Adaptation
```bash
python -m da4mt adapt --model <model> --train-file <trainfile> --output <outputdir> --normalization-file <name>_normalization_values.json --splits-file <name>_<i>.<splitter>_splits.json --hydra-config-name <config>
```
`<model>` should be the path to the directory containing the model, e.g. `outputdir/mlm-bert-30`
`trainfile` should be `<name>_mtr.jsonl` and additionally the normalization values need to be passed using `--normalization-file <name>_normalization_values.json`.
`<outputdir>` should be the directory were the weights of the domain adapted models are saved.
`<splits-file>` (optional) should be one of the split files created in the previous step.
`<config>` is the name of a config file in `../con` which contains the Hugging Face Trainer training arguments.
