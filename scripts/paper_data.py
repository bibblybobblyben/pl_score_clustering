"""
General one stop shop to create the data needed to produce plots for
publication

"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from pinpointlearning.model import LogReg, KNN
from pinpointlearning.utils import load_sample_data


#####
# Data description plots
######

with open("../data/Processed_Exams.json", "r", encoding="utf-8") as f:
    exam_metadata = json.load(f)

# How many students sit each exam, and how does cleaning change it?
n_students = []
n_pre = []
enames = []

frac_raw = {}
frac_bin = {}

mask = np.load("../data/outputs/CleanMask.npy")
for file in range(13):
    data = np.load(f"../data/processed/scores/Exam_{file}.npy")
    n_pre.append(sum(((~np.isnan(data)).sum(axis=1) > 0)))
    n_students.append(sum(mask * ((~np.isnan(data)).sum(axis=1) > 0)))
    enames.append(f"Exam {file}")
    # exam grades
    mask = np.load(f"../data/processed/masks/Exam_{file}_Mask.npy")
    binary_scores = np.load(f"../data/processed/binarised/Exam_{file}.npy")
    select = np.array(mask * ((~np.isnan(data)).sum(axis=1) > 0), dtype=bool)
    exam_meta = exam_metadata[file]
    poss_score = sum(exam_metadata[file]["maxes"])
    frac_raw[f"Exam_{file}"] = (np.sum(data, axis=1) / poss_score)[select].tolist()
    frac_bin[f"Exam_{file}"] = (np.sum(binary_scores, axis=1) / binary_scores.shape[1])[
        select
    ].tolist()


exam_scores = [{"raw": frac_raw, "binarised": frac_bin}]

with open("../data/outputs/TotalExamScores.json", "w", encoding="utf-8") as f:
    json.dump(exam_scores, f)


df = pd.DataFrame()
df["Exam"] = enames
df["Number"] = n_students
df["Pre"] = n_pre
df.to_csv("../data/outputs/StudentExamCountBreakdown.csv")


comp = np.zeros((13, 13))

# How many students sit each combination of exam?
for i in range(13):
    mask = np.load(f"../data/processed/masks/Exam_{i}_Mask.npy")
    ref = ~np.isnan(np.load(f"../data/processed/scores/Exam_{i}.npy")[:, 0])
    ref = mask * ref
    for j in range(13):
        mask2 = np.load(f"../data/processed/masks/Exam_{j}_Mask.npy")
        target = ~np.isnan(np.load(f"../data/processed/scores/Exam_{j}.npy")[:, 0])
        target = mask * target
        comp[i, j] = np.sum(target * ref)

np.save("../data/outputs/PaperCoincidences.npy", comp)
print(comp)


# how strongly related are scores in each question, for each exam?
allcorrs = []

allscores = []
for i in range(13):
    mask = np.load(f"../data/processed/masks/Exam_{i}_Mask.npy")
    data = np.load(f"../data/processed/binarised/Exam_{i}.npy")
    ref = ~np.isnan(data)[:, 0]
    itercorr = np.zeros((data.shape[1], data.shape[1]))
    data = np.array(data[ref, :], dtype=bool)
    nscores = []
    for q in range(data.shape[1]):
        for p in range(data.shape[1]):
            itercorr[q, p] = np.sum(data[:, q] * data[:, p]) / sum(
                data[:, q]
            )  # data.shape[0]
        nscores.append(float(sum(data[:, q]) / len(data[:, q])))
    allcorrs.append({f"Exam_{i}": itercorr.tolist()})
    allscores.append({f"Exam_{i}": nscores})

with open("../data/outputs/QuestionCorrMats.json", "w", encoding="utf-8") as f:
    json.dump(allcorrs, f)

with open("../data/outputs/QuestionScores.json", "w", encoding="utf-8") as f:
    json.dump(allscores, f)


# placeholder data ingestion
N_COLS = 23
data = load_sample_data(n_cols=N_COLS, n_rows=3 * 10**3, synthetic=False, binary=True)
print(data)

features = data[:, :N_COLS]  # [:, :-1]
target = data[:, -1]
target = target / np.amax(target)
target = np.array(target > 0.5, dtype=int).reshape(-1, 1)

traintest, val = train_test_split(data, train_size=0.9, random_state=2)
train, test = train_test_split(traintest, train_size=0.85, random_state=2)
###
# Explore how performance of KNN changes with number of neighbours
###


n_neighbours = list(range(2, 10, 2)) + [a * 10 for a in range(1, 10)]


by_target = []
# iterate over papers?
losses_knn = []
losses_lr = []
for qnum in range(N_COLS):
    losses = []
    colmask = [a != qnum for a in range(N_COLS)]
    features = train[:, colmask]
    target = np.array(train[:, qnum], dtype=int)
    test_features = test[:, colmask]
    test_target = np.array(test[:, qnum], dtype=int)
    test_ll = []
    for n in n_neighbours:
        clf = KNN(n_neighbours=n)
        clf.fit(features=features, target=target)
        probs = clf.predict_proba(features)
        losses.append(log_loss(target, probs[:, 1]))
        print(np.unique(test_target))
        test_ll.append(log_loss(test_target, clf.predict_proba(test_features)))
    n_chosen = n_neighbours[np.argwhere(np.array(test_ll) == np.amin(test_ll))[0][0]]

    clf = KNN(n_neighbours=n_chosen)
    clf.fit(features=features, target=target)
    losses_knn.append(log_loss(test_target, clf.predict_proba(test_features)))

    lr = LogReg()
    lr.fit(features=features, target=target)
    losses_lr.append(log_loss(test_target, lr.predict_proba(test_features)))

    by_target.append(losses)
results = {"n_neighbours": n_neighbours, "log_losses": by_target}
print(results)

with open(
    "../data/outputs/knn_n_neighbours_performance.json", "w", encoding="utf-8"
) as f:
    json.dump(results, f)


###
# Compare all models in a single plot
###

knn = KNN(n_neighbours=25)
knn.fit(features=features, target=target)
lr = LogReg()
lr.fit(features=features, target=target)
model_names = ["LogReg", "KNN"]
performances = [
    log_loss(target, lr.predict_proba(features)[:, 1]),
    log_loss(target, knn.predict_proba(features)[:, 1]),
]

results = {"model_names": model_names, "log_losses": performances}
with open("../data/outputs/model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(results, f)
