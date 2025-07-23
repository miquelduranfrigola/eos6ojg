from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Scaffolds import rdScaffoldNetwork


def get_scaffolds_text(smiles_list):

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

    murcko_scaffolds = scaffolds_smiles

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

    R = []
    for m, b in zip(murcko_scaffolds, brics_scaffolds):
        if m is None or m == "":
            r = []
        else:
            r = [m]
        r += b.split(",")
        R += [" ".join(r)]
    return R