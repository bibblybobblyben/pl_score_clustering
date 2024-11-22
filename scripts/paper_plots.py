"""
Produce all plots for the paper in one fell swoop. Simplifies making changes
to all plots at once.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import pinpointlearning as pl

###
# Compare all of the models on a single figure
###

if pl.plots_ready():
    print("Successfully set mplparams. Creating plots")
else:
    raise RuntimeError("Package has not imported.")

####
# Paper corr mat
####

data = np.load("../data/outputs/PaperCoincidences.npy")
for row in range(data.shape[0]):
    data[row, :] = data[row, :] / data[row, row]

fig, ax = plt.subplots(nrows=1, ncols=1)
cplot = ax.matshow(data, cmap="hccmap")
ax.set_xticks(
    list(range(data.shape[1])),
    labels=[f"Exam {a}" for a in range(data.shape[1])],
    rotation=90,
    size=6,
)
ax.set_yticks(
    list(range(data.shape[1])),
    labels=[f"Exam {a}" for a in range(data.shape[1])],
    size=6,
)
fig.colorbar(cplot, ax=ax, label="Fractional overlap")
ax.minorticks_off()
fig.tight_layout()
fig.savefig("../figs/PaperCorrelationPlot.png", dpi=500)


with open("../data/outputs/model_comparison.json", encoding="utf-8") as f:
    results = json.load(f)

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.scatter(results["model_names"], results["log_losses"])
ax.set_xlabel("Model")
ax.set_ylabel("Log loss")
fig.tight_layout()
fig.savefig("../figs/ModelComparison.png", dpi=250)


###
# View KNN performance as f(n_neighbours)
###

with open("../data/outputs/knn_n_neighbours_performance.json", encoding="utf-8") as f:
    results = json.load(f)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(results["n_neighbours"], np.array(results["log_losses"]).mean(axis=0))
ax.set_xlabel("Number of neighbours")
ax.set_ylabel("Log loss")
fig.tight_layout()
fig.savefig("../figs/KNN_N_neighbours.png", dpi=250)
