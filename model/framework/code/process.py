import os
import datamol as dm
from rdkit import Chem
from standardiser import standardise
from rdkit.Chem.Scaffolds import MurckoScaffold

root = os.path.join(os.path.dirname(__file__))

dm.disable_rdkit_log()

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
            return smiles_1
        else:
            print("Could not process with standardiser:", smiles_0)
            return smiles_0
    else:
        print("Could not process with datamol:", smiles)
        smiles_0 = preprocess_with_standardiser(smiles)
        if smiles_0 is not None:
            return smiles_0
        else:
            print("Could not process with standardiser either:", smiles)
            smiles_1 = preprocess_with_rdkit(smiles)
            if smiles_1 is not None:
                return smiles_1
            else:
                print("Could not process with rdkit either:", smiles)
                return None