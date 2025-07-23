import os
import datamol as dm
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import csv
from standardiser import standardise
from FPSim2.io import create_db_file

root = os.path.join(os.path.dirname(__file__))

dm.disable_rdkit_log()
DATASET_FILE = os.path.join(root, "..", "..", "checkpoints", "ADB_all_found_smiles.csv")

dest_dir = os.path.join(root, "..", "..", "checkpoints")

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
        

smiles_list = []
inchikey_list = []
with open(DATASET_FILE, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        smiles = r[0]
        smiles, inchikey = preprocess(smiles)
        if smiles is None:
            continue
        smiles_list += [smiles]
        inchikey_list += [inchikey]

data = {"smiles": smiles_list, "inchikey": inchikey_list}

df = pd.DataFrame(data)

df.drop_duplicates(subset=["smiles"], inplace=True)
df.drop_duplicates(subset=["inchikey"], inplace=True)
df = df.reset_index(drop=True)
print(df)
print(df.shape)

smiles_list = df["smiles"].tolist()
inchikey_list = df["inchikey"].tolist()

mols = [[smiles, i] for i, smiles in enumerate(smiles_list)]

print("Creating a database file with Morgan fingerprints")

create_db_file(
    mols_source=mols,
    filename=os.path.join(root, "..", "..", "checkpoints", "fpsim2_database.h5"),
    mol_format='smiles',
    fp_type='Morgan',
    fp_params={'radius': 2, 'fpSize': 1024}
)

with open(os.path.join(root, "..", "..", "checkpoints", "fpsim2_database_smiles.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles", "index"])
    for smiles, i in mols:
        writer.writerow([smiles, i])