"""
General one stop shop to create the data needed to produce plots for
publication

"""
import json
import numpy as np
from sklearn.metrics import log_loss
from pinpointlearning.model import LogReg, KNN
from pinpointlearning.utils import load_sample_data

# placeholder data ingestion
data = load_sample_data(n_cols=8, synthetic=False)
print(data)

features = data[:, :-1]
target = data[:, -1]
target = target / np.amax(target)
target = np.array(target > 0.5, dtype=int).reshape(-1, 1)


###
# Explore how performance of KNN changes with number of neighbours
###


n_neighbours = list(range(2, 10, 2)) + [a * 10 for a in range(1, 10)]

losses = []
for n in n_neighbours:
    clf = KNN(n_neighbours=n)
    clf.fit(features=features, target=target)
    probs = clf.predict_proba(features)
    losses.append(log_loss(target, probs[:, 1]))

results = {"n_neighbours": n_neighbours, "log_losses": losses}

with open("../data/outputs/knn_n_neighbours_performance.json", "w", encoding="rb") as f:
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
with open("../data/outputs/model_comparison.json", "w", encoding="rb") as f:
    json.dump(results, f)
