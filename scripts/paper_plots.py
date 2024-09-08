"""
Produce all plots for the paper in one fell swoop. Simplifies making changes
to all plots at once.
"""
import json
import matplotlib.pyplot as plt
import pinpointlearning as pl

###
# Compare all of the models on a single figure
###

if pl.plots_ready():
    print("Successfully set mplparams. Creating plots")
else:
    raise RuntimeError("Package has not imported.")

with open("../data/outputs/model_comparison.json", encoding="rb") as f:
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

with open("../data/outputs/knn_n_neighbours_performance.json", encoding="rb") as f:
    results = json.load(f)

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.scatter(results["n_neighbours"], results["log_losses"])
ax.set_xlabel("Number of neighbours")
ax.set_ylabel("Log loss")
fig.tight_layout()
fig.savefig("../figs/KNN_N_neighbours.png", dpi=250)
