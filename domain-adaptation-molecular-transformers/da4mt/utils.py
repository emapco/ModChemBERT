import json
import logging
import pathlib
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


class PhysicoChemcialPropertyExtractor:
    """Computes RDKit properties on-the-fly."""

    @staticmethod
    def get_surface_descriptor_subset() -> list[str]:
        """MOE-like surface descriptors  (Copied from the MolBERT paper)
        EState_VSA: VSA (van der Waals surface area) of atoms contributing to a specified bin of e-state
        SlogP_VSA: VSA of atoms contributing to a specified bin of SlogP
        SMR_VSA: VSA of atoms contributing to a specified bin of molar refractivity
        PEOE_VSA: VSA of atoms contributing to a specified bin of partial charge (Gasteiger)
        LabuteASA: Labute's approximate surface area descriptor
        """
        return [
            "SlogP_VSA1",
            "SlogP_VSA10",
            "SlogP_VSA11",
            "SlogP_VSA12",
            "SlogP_VSA2",
            "SlogP_VSA3",
            "SlogP_VSA4",
            "SlogP_VSA5",
            "SlogP_VSA6",
            "SlogP_VSA7",
            "SlogP_VSA8",
            "SlogP_VSA9",
            "SMR_VSA1",
            "SMR_VSA10",
            "SMR_VSA2",
            "SMR_VSA3",
            "SMR_VSA4",
            "SMR_VSA5",
            "SMR_VSA6",
            "SMR_VSA7",
            "SMR_VSA8",
            "SMR_VSA9",
            "EState_VSA1",
            "EState_VSA10",
            "EState_VSA11",
            "EState_VSA2",
            "EState_VSA3",
            "EState_VSA4",
            "EState_VSA5",
            "EState_VSA6",
            "EState_VSA7",
            "EState_VSA8",
            "EState_VSA9",
            "LabuteASA",
            "PEOE_VSA1",
            "PEOE_VSA10",
            "PEOE_VSA11",
            "PEOE_VSA12",
            "PEOE_VSA13",
            "PEOE_VSA14",
            "PEOE_VSA2",
            "PEOE_VSA3",
            "PEOE_VSA4",
            "PEOE_VSA5",
            "PEOE_VSA6",
            "PEOE_VSA7",
            "PEOE_VSA8",
            "PEOE_VSA9",
            "TPSA",
        ]

    def __init__(self, logger, subset="all"):
        super().__init__()

        # Ipc takes on extremly large values for some molecules 10^10 - 10^195 in e.g. the sider and freesolv datasets
        # leading to inf values during e.g. standard deviation calulation and  during predictions, completely
        # throwing of the models
        forbidden = set(["Ipc"])

        if subset == "all":
            self.descriptors = [name for name, _ in Chem.Descriptors.descList if name not in forbidden]
        elif subset == "surface":
            self.descriptors = self.get_surface_descriptor_subset()

        self.calculator = MolecularDescriptorCalculator(self.descriptors)
        self.num_labels = len(self.descriptors)
        logger.info(f"Number of physicochemical properties: {self.num_labels}")

        assert all(f not in self.descriptors for f in forbidden), "Invalid descriptors encountered"

    def compute_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_descriptors = np.full(shape=(self.num_labels), fill_value=0.0)
        else:
            mol_descriptors = np.array(list(self.calculator.CalcDescriptors(mol)))
            mol_descriptors = np.nan_to_num(mol_descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        assert mol_descriptors.size == self.num_labels

        return mol_descriptors

    def compute_batch(self, smiles: list[str], n_jobs: int = 25) -> tuple[str, np.ndarray]:
        """
        Computes the physicochemcical properties of all SMILES strings in the list
        in parallel.
        :param smiles: list of molecules in their SMILES representation
        :return: tuple with list of smiles and a list with the corresponding properties
        """
        # Calculate the properties in parallel
        physicochemical_fingerprints = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.compute_descriptors)(smi) for smi in smiles
        )

        return smiles, physicochemical_fingerprints


def extract_physicochemical_props(
    smiles: list[str],
    output_path: pathlib.Path,
    logger: logging.Logger,
    normalization_path: pathlib.Path = None,
    subset: Literal["all", "surface"] = "all",
):
    """Extracts physicochemical properties from the dataset and saves them to a JSONL file.

    Args:
        smiles (pathlib.Path): Path to the dataset.
        output_path (pathlib.Path): Path to save the labeled dataset.
        normalization_path (pathlib.Path, optional): Path to save the normalization values (mean and std).
    """
    extractor = PhysicoChemcialPropertyExtractor(logger, subset=subset)

    smiles, physicochemical_fingerprints = extractor.compute_batch(smiles)

    logger.info(f"Finished computing physicochemical properties for {len(physicochemical_fingerprints)} molecules")

    # Save the properties in JSONL format
    with open(output_path, "w") as labeled_dataset_file:
        for smile, physicochemical_property in zip(smiles, physicochemical_fingerprints, strict=False):
            json.dump(
                {"smile": smile, "labels": physicochemical_property.tolist()},
                labeled_dataset_file,
            )
            labeled_dataset_file.write("\n")
    logger.info(f"Saved labeled dataset to {output_path}")

    if normalization_path:
        # Compute mean and std
        prop_arr = np.array(physicochemical_fingerprints)
        mean = np.mean(prop_arr, axis=0)
        std = np.std(prop_arr, axis=0)

        logger.info(
            f"Computed normalization values (mean and std) of {len(physicochemical_fingerprints[0])}                   "
            f"  physicochemical properties. Number of molecules: {len(prop_arr)}"
        )

        # dump the output as jsonl to be used for in pre-training
        with open(normalization_path, "w") as normalization_file:
            json.dump(
                {
                    "mean": list(mean),
                    "std": list(std),
                    "label_names": extractor.descriptors,
                },
                normalization_file,
            )
        logger.info(f"Saved normalization values to {normalization_path}")
