import os
import csv
import random
import collections
import joblib
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB as NaiveBayes
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_curve

root = os.path.join(os.path.dirname(__file__))

words_pos = collections.defaultdict(list)

print("Reading ADB scaffolds")
adb = []
with open(os.path.join(root, "..", "..", "checkpoints", "adb_murcko_scaffolds.tsv"), "r") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    for r in reader:
        sc = r[2]
        if sc is None or sc == "":
            adb += [[]]
        else:
            adb += [[sc]]

print("Reading ADB BRICS scaffolds")
with open(os.path.join(root, "..", "..", "checkpoints", "adb_brics_scaffolds.tsv"), "r") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    for i, r in enumerate(reader):
        sc = r[1]
        if sc is None or sc == "":
            continue
        else:
            adb[i] += sc.split(",")

print("Concatenating ADB scaffolds")
adb = [" ".join(sc) for sc in adb if len(sc) > 0]

print("Building the tokenizer")
vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None, min_df=3)

vectorizer.fit(adb)
Xa = vectorizer.transform(adb)

print(Xa.shape)

print("Reading ChEMBL scaffolds")
chembl = []
with open(os.path.join(root, "..", "..", "checkpoints", "chembl_murcko_scaffolds.tsv"), "r") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    for r in reader:
        sc = r[2]
        if sc is None or sc == "":
            chembl += [[]]
        else:
            chembl += [[sc]]

print("Reading ChEMBL BRICS scaffolds")
with open(os.path.join(root, "..", "..", "checkpoints", "chembl_brics_scaffolds.tsv"), "r") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    for i, r in enumerate(reader):
        sc = r[1]
        if sc is None or sc == "":
            continue
        else:
            chembl[i] += sc.split(",")

chembl = [" ".join(sc) for sc in chembl if len(sc) > 0]

random.shuffle(chembl)

print("Transforming the data...")
Xc = vectorizer.transform(chembl)

print(Xc.shape)

from scipy.sparse import vstack
import numpy as np

X = vstack([Xa, Xc])
y = [1] * Xa.shape[0] + [0] * Xc.shape[0]
y = np.array(y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

# To gather probs and labels for threshold tuning
all_probs = []
all_true = []

# Cross-validation loop
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    score = balanced_accuracy_score(y_test, y_pred)
    scores.append(score)

    all_probs.extend(y_prob)
    all_true.extend(y_test)

print(f"Mean balanced accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
print(f"AUROC")

all_probs = np.array(all_probs)
all_true = np.array(all_true)

# ROC curve gives FPR, TPR, thresholds
fpr, tpr, thresholds = roc_curve(all_true, all_probs)
from sklearn.metrics import auc
import matplotlib.pyplot as plt
print("AUROC:", auc(fpr, tpr))

# Youden index = TPR - FPR
youden_index = tpr - fpr
best_threshold = thresholds[np.argmax(youden_index)]

print(f"Best threshold (Youden index): {best_threshold:.3f}")

data = {
    "best_threshold": best_threshold,
    "roc_auc": auc(fpr, tpr)
}

clf = NaiveBayes()
clf.fit(X, y)

vectorizer_file = os.path.join(root, "..", "..", "checkpoints", "vectorizer.joblib")
model_file = os.path.join(root, "..", "..", "checkpoints", "model.joblib")
data_file = os.path.join(root, "..", "..", "checkpoints", "data.json")

with open(data_file, "w") as f:
    json.dump(data, f, indent=4)

joblib.dump(vectorizer, vectorizer_file)
joblib.dump(clf, model_file)