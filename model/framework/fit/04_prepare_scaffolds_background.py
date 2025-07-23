import os
import csv
import pandas as pd
from tqdm import tqdm
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Scaffolds import rdScaffoldNetwork
from rdkit.Chem import Descriptors

root = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(root, "..", "..", "checkpoints", "chembl_smiles.csv")

smiles_list = []
with open(file_name, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)
    for r in tqdm(reader, desc="Reading SMILES"):
        smiles_list += [r[0]]

smiles_list = list(set(smiles_list))

random.shuffle(smiles_list)

smiles_list = [smi for smi in tqdm(smiles_list, desc="Filtering by MWt < 1000") if Descriptors.MolWt(Chem.MolFromSmiles(smi)) < 1000]

print("Number of unique SMILES:", len(smiles_list))

print("Getting Murcko scaffolds")

scaffolds = []
for smiles in tqdm(smiles_list, desc="Generating Murcko scaffolds"):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffolds += [Chem.MolToSmiles(scaffold)]

print("Converting Murcko scaffolds to InChIKeys and SMILES")

scaffolds_inchikeys = []
scaffolds_smiles = []
for smi, scaffold in tqdm(zip(smiles_list, scaffolds), desc="Evaluating scaffolds"):
    mol = Chem.MolFromSmiles(scaffold)
    if mol is None:
        scaffolds_inchikeys += [None]
        scaffolds_smiles += [None]
        continue
    inchikey = Chem.inchi.MolToInchiKey(mol)
    if scaffold is None or scaffold == "":
        scaffolds_inchikeys += [None]
        scaffolds_smiles += [None]
        continue
    if inchikey is None or inchikey == "":
        scaffolds_inchikeys += [None]
        scaffolds_smiles += [None]
        continue
    scaffold = Chem.MolToSmiles(mol, canonical=True)
    scaffolds_inchikeys += [inchikey]
    scaffolds_smiles += [scaffold]

assert len(scaffolds_inchikeys) == len(smiles_list)
assert len(scaffolds_smiles) == len(smiles_list)

data = {"original_smiles": smiles_list, "inchikey": scaffolds_inchikeys, "murcko_scaffold": scaffolds_smiles}

df = pd.DataFrame(data)
df.to_csv(os.path.join(root, "..", "..", "checkpoints", "chembl_murcko_scaffolds.tsv"), index=False, sep="\t")

print("Done with Murcko scaffolds")

print("Working on BRICS-based scaffold network...")

def get_brics_scaffolds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    options = rdScaffoldNetwork.ScaffoldNetworkParams()
    options.includeGenericScaffolds = True
    options.includeScaffoldsWithAttachments = False
    options.bricsLabels = True
    network = rdScaffoldNetwork.CreateScaffoldNetwork([mol], options)
    scaffolds = []
    for node in network.nodes:
        mol = Chem.MolFromSmiles(node)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        scaffolds += [smiles]
    return scaffolds

R = []
brics_scaffolds = []
for smiles in tqdm(smiles_list, desc="Generating BRICS scaffolds"):
    try:
        brics = get_brics_scaffolds(smiles)
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        brics_scaffolds += [None]
        continue
    if brics is None:
        brics_scaffolds += [None]
        continue
    brics_scaffolds += [",".join(brics)]

assert len(brics_scaffolds) == len(smiles_list)

with open(os.path.join(root, "..", "..", "checkpoints", "chembl_brics_scaffolds.tsv"), "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["original_smiles", "brics_scaffolds"])
    for smiles, brics in zip(smiles_list, brics_scaffolds):
        writer.writerow([smiles, brics])