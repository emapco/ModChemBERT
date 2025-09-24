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

import warnings

import datasets
import numpy as np
from rdkit import Chem


def normalization(ds: datasets.arrow_dataset.Dataset) -> datasets.arrow_dataset.Dataset:
    """Standardize float label columns using mean and std (DeepChem-style).

    Implements y' = (y - mean) / std with nan -> 0 handling similar to
    np.nan_to_num. Only float-typed columns are transformed (labels), and
    string/int feature columns (e.g., SMILES) are left untouched.
    """

    # Identify float-valued columns (skip ints/strings)
    float_cols: list[str] = []
    for col, feat in ds.features.items():
        dtype = getattr(feat, "dtype", None)
        if isinstance(dtype, str) and dtype.startswith("float"):
            float_cols.append(col)

    if not float_cols:
        return ds

    # Compute column-wise means and stds ignoring NaNs
    stats: dict[str, tuple[float, float]] = {}
    for col in float_cols:
        values = ds[col]
        arr = np.array([float(v) if v is not None else np.nan for v in values], dtype=np.float64)
        mean = float(np.nanmean(arr)) if arr.size > 0 else 0.0
        std = float(np.nanstd(arr)) if arr.size > 0 else 1.0
        # Avoid divide-by-zero or nan std
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        stats[col] = (mean, std)

    def _normalize_batch(batch):
        import math

        for col in float_cols:
            mean, std = stats[col]
            vals = batch[col]
            out: list[float] = []
            for v in vals:
                norm_v = float("nan") if v is None else (float(v) - mean) / std
                if not math.isfinite(norm_v):
                    norm_v = 0.0
                out.append(norm_v)
            batch[col] = out
        return batch

    return ds.map(_normalize_batch, batched=True)


def log(ds: datasets.arrow_dataset.Dataset) -> datasets.arrow_dataset.Dataset:
    """Apply a log1p transform to float columns (labels).

    Since features are SMILES strings (non-numeric), this targets numeric float
    columns only, which correspond to labels in our datasets.
    """

    # Identify float-valued columns (skip ints/strings)
    float_cols: list[str] = []
    for col, feat in ds.features.items():
        dtype = getattr(feat, "dtype", None)
        if isinstance(dtype, str) and dtype.startswith("float"):
            float_cols.append(col)

    if not float_cols:
        return ds

    def _log_batch(batch):
        import math

        for col in float_cols:
            values = batch[col]
            # Handle None/missing values gracefully
            batch[col] = [None if v is None else math.log1p(float(v)) for v in values]
        return batch

    return ds.map(_log_batch, batched=True)


def randomize_smiles(smiles: str, rng: np.random.Generator, isomeric=True) -> str:
    """Creates a random enumeration of this molecules SMILES representation if possible.

    Args:
        smiles (str): Molecule in SMILES representation
        rng (np.random.Generator): (seeded) PRNG
        isomeric (bool): Include stereochemistry information, default to True

    Returns:
        str: Random enumeration of this molecules smiles string
    """
    mol = Chem.MolFromSmiles(smiles)
    enumerations = set(
        Chem.MolToRandomSmilesVect(
            mol,
            numSmiles=100,
            isomericSmiles=isomeric,
            randomSeed=int(rng.integers(0, 1000)),
        )
    )

    # The random enumeration may include the original string as well
    if smiles in enumerations:
        if len(enumerations) == 1:
            warnings.warn(f"{smiles} most likely can't be randomized", stacklevel=2)
            return smiles
        # Pick one of the other options
        enumerations.remove(smiles)
        return rng.choice(list(enumerations))

    # Pick any of the enumerations
    return rng.choice(list(enumerations))
