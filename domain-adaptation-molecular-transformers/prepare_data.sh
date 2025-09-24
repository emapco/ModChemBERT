ADME_DIR="data/adme"
ASTRA_DIR="data/astrazeneca"
COMBINED="data/combined"
DATA_DIR="data"

# ADME
python -m da4mt prepare dataset $ADME_DIR/adme_microsom_stab_h_cleaned.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_microsom_stab_r_cleaned.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_permeability.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_ppb_h.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_ppb_r.csv -o $DATA_DIR
python -m da4mt prepare dataset $ADME_DIR/adme_solubility.csv -o $DATA_DIR
# Astrazeneca
python -m da4mt prepare dataset $ASTRA_DIR/astrazeneca_CL.csv -o $DATA_DIR
python -m da4mt prepare dataset $ASTRA_DIR/astrazeneca_LogD74.csv -o $DATA_DIR
python -m da4mt prepare dataset $ASTRA_DIR/astrazeneca_PPB.csv -o $DATA_DIR
python -m da4mt prepare dataset $ASTRA_DIR/astrazeneca_Solubility.csv -o $DATA_DIR
# Combined
python -m da4mt prepare dataset $COMBINED/all_datasets_smiles.csv -o $DATA_DIR

# Splitting
python -m da4mt prepare splits $ADME_DIR/adme_microsom_stab_h_cleaned.csv -o $DATA_DIR --splitter scaffold --num-splits 1
python -m da4mt prepare splits $ADME_DIR/adme_microsom_stab_r_cleaned.csv -o $DATA_DIR --splitter scaffold --num-splits 1
python -m da4mt prepare splits $ADME_DIR/adme_permeability.csv -o $DATA_DIR --splitter scaffold --num-splits 1
python -m da4mt prepare splits $ADME_DIR/adme_ppb_h.csv -o $DATA_DIR --splitter scaffold --num-splits 1
python -m da4mt prepare splits $ADME_DIR/adme_ppb_r.csv -o $DATA_DIR --splitter scaffold --num-splits 1
python -m da4mt prepare splits $ADME_DIR/adme_solubility.csv -o $DATA_DIR --splitter scaffold --num-splits 1

python -m da4mt prepare splits $ASTRA_DIR/astrazeneca_CL.csv -o $DATA_DIR --splitter scaffold --num-splits 1
python -m da4mt prepare splits $ASTRA_DIR/astrazeneca_LogD74.csv -o $DATA_DIR --splitter scaffold --num-splits 1
python -m da4mt prepare splits $ASTRA_DIR/astrazeneca_PPB.csv -o $DATA_DIR --splitter scaffold --num-splits 1
python -m da4mt prepare splits $ASTRA_DIR/astrazeneca_Solubility.csv -o $DATA_DIR --splitter scaffold --num-splits 1

python -m da4mt prepare splits $COMBINED/all_datasets_smiles.csv -o $DATA_DIR --splitter scaffold --num-splits 1
