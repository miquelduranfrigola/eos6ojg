import os
import csv
from tqdm import tqdm
from rdkit import Chem
import datamol as dm
from standardiser import standardise

dm.disable_rdkit_log()

root = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(root, "..", "..", "checkpoints", "chembl_35_chemreps.txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    print(header)
    smiles_list = [r[1] for r in reader]


def preprocess_with_datamol(smiles):
    mol = dm.to_mol(smiles)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol)
    smiles = dm.to_smiles(mol)
    return smiles

def preprocess_with_standardiser(smiles):
    mol = Chem.MolFromSmiles(smiles)
    try:
        mol = standardise.run(mol)
    except:
        mol = None
    if mol is None:
        return None
    smiles = Chem.MolToSmiles(mol)
    return smiles

def preprocess_with_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    smiles = Chem.MolToSmiles(mol)
    return smiles

def get_inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchikey = Chem.MolToInchiKey(mol)
    return inchikey

def preprocess(smiles):
    smiles_0 = preprocess_with_datamol(smiles)
    if smiles_0 is not None:
        smiles_1 = preprocess_with_standardiser(smiles_0)
        if smiles_1 is not None:
            return smiles_1, get_inchikey(smiles_1)
        else:
            print("Could not process with standardiser:", smiles_0)
            return smiles_0, get_inchikey(smiles_0)
    else:
        print("Could not process with datamol:", smiles)
        smiles_0 = preprocess_with_standardiser(smiles)
        if smiles_0 is not None:
            return smiles_0, get_inchikey(smiles_0)
        else:
            print("Could not process with standardiser either:", smiles)
            smiles_1 = preprocess_with_rdkit(smiles)
            if smiles_1 is not None:
                return smiles_1, get_inchikey(smiles_1)
            else:
                print("Could not process with rdkit either:", smiles)
                return None, None

processed_smiles_list = []
for smiles in tqdm(smiles_list):
    processed_smiles, inchikey = preprocess(smiles)
    if processed_smiles is not None:
        processed_smiles_list += [processed_smiles]
    else:
        print(f"Failed to process SMILES: {smiles}")

with open(os.path.join(root, "..", "..", "checkpoints", "chembl_smiles.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])
    for smiles in processed_smiles_list:
        writer.writerow([smiles])