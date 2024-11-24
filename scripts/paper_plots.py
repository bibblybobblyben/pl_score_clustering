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
cplot = ax.matshow(data, cmap="hccmap_lblues", vmin=0, vmax=1)
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


#####
# What do the scores of students typically look like?
#####

with open("../data/outputs/TotalExamScores.json", "r", encoding="utf-8") as f:
    tot_scores = json.load(f)[0]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 4))
labels = []
means = []
stds = []
errs = []
allraws = []
allbins = []
for i in range(len(tot_scores["raw"])):
    labels.append(f"Exam {i}")
    _ = [allraws.append(a) for a in tot_scores["raw"][f"Exam_{i}"]]
    _ = [allbins.append(a) for a in tot_scores["binarised"][f"Exam_{i}"]]
    means.append(np.mean(tot_scores["binarised"][f"Exam_{i}"]))
    stds.append(
        np.std(tot_scores["binarised"][f"Exam_{i}"])
        # / np.sqrt(len(tot_scores["binarised"][f"Exam_{i}"]))
    )
    errs.append(
        np.std(tot_scores["binarised"][f"Exam_{i}"])
        / np.sqrt(len(tot_scores["binarised"][f"Exam_{i}"]))
    )


ax[0].errorbar(
    labels,
    means,
    yerr=stds,
    color="hclightblue",
    marker="",
    linestyle=" ",
    linewidth=0,
    capsize=8,
    capthick=0.8,
    elinewidth=0.5,
)
ax[0].scatter(labels, means, marker="o", color="hclightblue")
ax[0].set_ylim((0, 1))
ax[0].set_ylabel("Fractional score")
ax[0].set_title("Score distributions")
ax[0].tick_params(axis="x", labelrotation=90)

ax[1].errorbar(
    labels,
    means,
    yerr=errs,
    color="hccoral",
    marker="",
    linestyle=" ",
    linewidth=0,
    capsize=8,
    capthick=0.8,
    elinewidth=0.5,
)
ax[1].scatter(labels, means, marker="o", color="hccoral")
ax[1].tick_params(axis="x", labelrotation=90)
ax[1].set_ylabel("Fractional score")
ax[1].set_title("Mean scores")
fig.tight_layout()
fig.savefig("../figs/StudentTypicalScores_ByExam.png", dpi=500)


bins = np.linspace(0, 1, 20)
fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(4, 4))

rdat = np.array(allraws).flatten()
bdat = np.array(allbins).flatten()
ax.hist(rdat, bins=bins, label="Raw")
ax.hist(bdat, bins=bins, histtype="step", linewidth=3, label="Binary")
ax.legend(loc="upper right")
ax.set_ylabel("Number of exams")
ax.set_xlabel("Fractional Grade")
ax.set_xlabel("Fractional Grade")


fig.savefig("../figs/DistributionOfExamScores.png", dpi=500)

####
# Relationship between scores in each exam
####

with open("../data/outputs/QuestionCorrMats.json", "r", encoding="utf-8") as f:
    data = json.load(f)
fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(10, 15))
ax = ax.ravel()

for i, exam in enumerate(data):
    for title, mat in exam.items():
        cba = ax[i].matshow(np.array(mat)[:20, :20], cmap="hccmap", vmin=0, vmax=1)
        ax[i].set_title(f"Exam {i}")
        ax[i].set_ylabel("q")
        ax[i].set_xlabel("p")
plt.colorbar(cba, ax=ax[14], label="Fraction", orientation="horizontal")
ax[14].axis("off")
fig.delaxes(ax[13])
fig.tight_layout()
fig.savefig("../figs/QuestionCorrMat.png", dpi=500)

#####
# Scores by question by exam
#####

with open("../data/outputs/QuestionScores.json", "r", encoding="utf-8") as f:
    data = json.load(f)
fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(15, 10))
ax = ax.ravel()

for i, exam in enumerate(data):
    for title, mat in exam.items():
        roll_ave = [np.mean(mat[max(0, j - 4) : j + 1]) for j in range(len(mat))]
        ax[i].bar(x=np.arange(1, 21), height=mat[:20])
        ax[i].plot(np.arange(1, 21), roll_ave[:20], c="hccoral", linewidth=6)
        ax[i].plot()
        ax[i].set_title(f"Exam {i}")
        ax[i].set_ylabel("Average score")
        ax[i].set_xlabel("Question")
        ax[i].set_ylim((0, 1))
# plt.colorbar(cba, ax=ax[14], label="Fraction", orientation="horizontal")
# ax[14].axis("off")
fig.delaxes(ax[13])
fig.delaxes(ax[14])
fig.tight_layout()
fig.savefig("../figs/QuestionScores.png", dpi=500)


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
