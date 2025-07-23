import os
import csv
from rdkit import Chem


root = os.path.dirname(os.path.abspath(__file__))

print(root)

db_file = os.path.join(root, "..", "..", "checkpoints", "ADB_all_compounds.csv")

with open(db_file, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)
    n_cols = len(header)
    all_found_smiles = []
    for i, row in enumerate(reader):
        row = row
        for r in row:
            if r == "":
                continue
            mol = Chem.MolFromSmiles(r)
            if mol is None:
                continue
            if mol.GetNumAtoms() < 10:
                continue
            all_found_smiles += [Chem.MolToSmiles(mol, isomericSmiles=True)]

all_found_smiles = list(set(all_found_smiles))

with open(os.path.join(root, "..", "..", "checkpoints", "ADB_all_found_smiles.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])
    for smiles in all_found_smiles:
        writer.writerow([smiles])
        