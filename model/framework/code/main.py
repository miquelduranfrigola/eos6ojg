import os
import sys
import csv
import joblib
import json
import numpy as np
from FPSim2 import FPSim2Engine
from tqdm import tqdm
import pandas as pd
import multiprocessing

NUM_CPU = max(1, int(multiprocessing.cpu_count() / 2))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
from process import preprocess
from get_scaffolds import get_scaffolds_text

checkpoints_dir = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))

input_file = sys.argv[1]
output_file = sys.argv[2]

smiles_list = []
with open(input_file, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        smiles = preprocess(r[0])
        if smiles:
            smiles_list += [smiles]

print("Processed SMILES in input:", len(smiles_list))

print("Loading FPSim2 database...")
fp_database = os.path.join(checkpoints_dir, "fpsim2_database.h5")
fpe = FPSim2Engine(fp_database, in_memory_fps=True)

print("Counting similarities...")
SIM_THRESHOLDS = [0.3, 0.5, 0.7, 0.9, 1.0]
C = np.zeros((len(smiles_list), len(SIM_THRESHOLDS)), dtype=int)
for j, smiles in tqdm(enumerate(smiles_list)):
    results = fpe.similarity(
        smiles,
        metric="tanimoto",
        threshold=min(SIM_THRESHOLDS),
        n_workers=NUM_CPU
    )
    counts = [0]*len(SIM_THRESHOLDS)
    for i, sim_threshold in enumerate(SIM_THRESHOLDS):
        c = 0
        for r in results:
            if r[1] >= sim_threshold:
                c += 1
        counts[i] = c
    C[j, :] = counts

cols = [f"num_sim_{thresh}".replace(".", "_") for thresh in SIM_THRESHOLDS]

df = pd.DataFrame(C, columns=cols)

print("Predicting the assignment based on Scaffolds")

vectorizer_file = os.path.join(root, "..", "..", "checkpoints", "vectorizer.joblib")
model_file = os.path.join(root, "..", "..", "checkpoints", "model.joblib")
data_file = os.path.join(root, "..", "..", "checkpoints", "data.json")

with open(data_file, "r") as f:
    data = json.load(f)

threshold = data["best_threshold"]

vectorizer = joblib.load(vectorizer_file)
model = joblib.load(model_file)

texts = get_scaffolds_text(smiles_list)
X = vectorizer.transform(texts)
probas = model.predict_proba(X)[:,1]
y_hat = []
for proba in probas:
    if proba >= threshold:
        y_hat += [1]
    else:
        y_hat += [0]

df["scaff_class"] = y_hat

df.to_csv(output_file, index=False)