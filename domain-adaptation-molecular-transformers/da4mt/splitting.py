import math
from collections import defaultdict
from typing import TypedDict

import numpy as np


class DatasetSplit(TypedDict):
    train: list[int]
    val: list[int]
    test: list[int]


class SplittingError(Exception): ...


def assert_no_overlap(folds: list[DatasetSplit]) -> None:
    """
    Assert that no overlap between val and test sets exists in different folds.
    :param folds: list of DatasetSplit objects
    """
    # Turns this function off if pythons optimized mode is activated
    # which would deactivate the assert statements anyway
    if not __debug__:
        return

    for i, a in enumerate(folds):
        for j, b in enumerate(folds):
            if i == j:
                continue

            for key in a:
                if key == "train":
                    continue

                overlap = set(a[key]) & set(b[key])
                assert len(overlap) == 0, f"Overlap between {key} sets in fold {i} and {j}: {overlap}"
                assert len(a[key]) == len(b[key]), f"Different sized splits: {key}, {i}, {j}"


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str | None:
    """
    Generate a Murcko scaffold from a SMILES string.

    :param str smiles: SMILES string of the molecule
    :param bool include_chirality: Whether to include chirality in the scaffold
    :return: Murcko scaffold SMILES string
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ModuleNotFoundError as e:
        raise ImportError("This function requires RDKit to be installed.") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def generate_scaffold_ds(smiles_list: list[str]) -> dict[str, list[int]]:
    """
    Generate a dictionary mapping scaffolds to molecule indices.

    :param smiles_list: list of SMILES strings
    :return: dictionary mapping scaffolds to lists of molecule indices
    """
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        if scaffold is not None:
            scaffolds[scaffold].append(ind)

    return {key: sorted(value) for key, value in scaffolds.items()}


def k_fold_scaffold_split(smiles: list[str], n: int) -> list[DatasetSplit]:
    """
    Attempts to split the dataset in to k-folds of train/test/validation set where the splits are
    determined by the scaffolds present in the dataset.
    This may fail, if not enough scaffolds with sufficient number of molecules are available
    without overlap between the test/val in the different folds.

    The algorithms roughly works as follows:
        Given a mapping from scaffolds to molecules which contain this molecule

        a) Iterate over all scaffolds
        b) If scaffold has been *used* as validation or test set, pack into training set
        c) Otherwise:
            1) Attempt to pack all molecules with this scaffold into the validation set
            2) Otherwise: Attempt to pack into the test set
            3) Otherwise: Pack into the training set

            4) If packed into validation or test set, mark this scaffold as 'used'
        d) Repeat k times.

    :param smiles: list of SMILES strings
    :param n: Number of folds
    :return: list of DatasetSplit objects, one for each fold
    """
    test_frac = 0.1
    val_frac = 0.1

    test_cutoff = math.floor(test_frac * len(smiles))
    val_cutoff = math.floor(val_frac * len(smiles))
    scaffolds = generate_scaffold_ds(smiles)

    folds: list[DatasetSplit] = [{"test": [], "val": [], "train": []} for _ in range(n)]

    test_or_val = [[False] * len(scaffolds) for _ in range(n)]

    for k in range(n):
        for i, (scaffold, mol_indices) in enumerate(scaffolds.items()):
            # If used as validation/test set in any fold before, this has to go into the
            # training set of this fold
            has_been_test_or_val = any(test_or_val[j][i] for j in range(n))
            if has_been_test_or_val:
                folds[k]["train"].extend(mol_indices)
            else:
                if len(mol_indices) + len(folds[k]["val"]) <= val_cutoff:
                    folds[k]["val"].extend(mol_indices)
                    test_or_val[k][i] = True
                elif len(mol_indices) + len(folds[k]["test"]) <= test_cutoff:
                    folds[k]["test"].extend(mol_indices)
                    test_or_val[k][i] = True
                else:
                    # This scaffold has too many molecules for test/validation set
                    # and needs to be assigned to the training set.
                    folds[k]["train"].extend(mol_indices)

        if len(folds[k]["test"]) < test_cutoff or len(folds[k]["val"]) < val_cutoff:
            raise SplittingError("Insufficient number of scaffolds to assign into val/test set without overlap.")

    assert_no_overlap(folds)
    return folds


def random_split(smiles: list[str], n: int, seed: int) -> list[DatasetSplit]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(smiles))
    rng.shuffle(indices)
    splits = np.array_split(indices, n)

    folds = []
    for split in range(n):
        val_split = split
        test_split = (split + 1) % n

        folds.append(
            {
                "train": np.concatenate([splits[k] for k in range(n) if k != val_split and k != test_split]).tolist(),
                "val": splits[val_split].tolist(),
                "test": splits[test_split].tolist(),
            }
        )

    return folds
