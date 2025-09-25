"""
Produce all plots for the paper in one fell swoop. Simplifies making changes
to all plots at once.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import pinpointlearning as pl
from math import ceil


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

# with open("../data/outputs/knn_n_neighbours_performance.json", encoding="utf-8") as f:
#    results = json.load(f)

# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.scatter(results["n_neighbours"], np.array(results["log_losses"]).mean(axis=0))
# ax.set_xlabel("Number of neighbours")
# ax.set_ylabel("Log loss")
# fig.tight_layout()
# fig.savefig("../figs/KNN_N_neighbours.png", dpi=250)


###
# View knn performance by exam
###

fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(15, 8))
ax = ax.ravel()
for i in range(13):
    with open(
        f"../data/outputs/fits/knn_n_neighbours_performance_{i}.json", encoding="utf-8"
    ) as f:
        results = json.load(f)
    losses = np.array(results["losses_by_exam"])
    xs = np.array(results["n_explored"])
    print(losses[i, :, :, :].shape)
    mus = np.mean(losses[i, :, :, :], axis=(0, 1))
    sigs = np.std(
        losses[i, :, :, :], axis=(0, 1)
    )  # / np.sqrt( losses.shape[1]*losses.shape[2] )
    ax[i].errorbar(
        xs,
        mus,
        yerr=sigs,
        # color="hccoral",
        marker="o",
        linestyle=" ",
        linewidth=0,
        capsize=8,
        capthick=0.8,
        elinewidth=0.5,
    )
    ax[i].set_xlabel("Number of neighbours")
    ax[i].set_ylabel("Log loss")
    ax[i].set_title(f"Exam {i}")
fig.delaxes(ax[13])
fig.delaxes(ax[14])
fig.tight_layout()
fig.savefig("../figs/KNNPerformance_by_exam.png", dpi=300)


###
# how does the exam chosen number of clusters change
###


fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(15, 8))
ax = ax.ravel()

for i in range(13):
    with open(
        f"../data/outputs/fits/knn_n_neighbours_performance_{i}.json", encoding="utf-8"
    ) as f:
        results = json.load(f)
    losses = np.array(results["n_neighbours_by_exam"])
    xs, mus = np.unique(losses[i], return_counts=True)
    ax[i].bar(
        xs,
        mus,
    )
    ax[i].set_xlabel("Optimal number of neighbours")
    ax[i].set_ylabel("Frequency")
    ax[i].set_title(f"Exam {i}")
    ax[i].set_xlim((0, np.amax(losses) + 1))
fig.delaxes(ax[13])
fig.delaxes(ax[14])
fig.tight_layout()
fig.savefig("../figs/KNN_nchosen_by_exam.png", dpi=300)

##How does each model do on test data - log loss?

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 16), dpi=300)
ax = ax.ravel()
for pnum in range(13):
    with open(
        f"../data/outputs/fits/paper_model_fitting_results_{pnum}.json",
        encoding="utf-8",
    ) as f:
        results = json.load(f)
    df = results["AllQuestions_test_performance"]
    ys = [
        np.mean(np.array(df["log_reg"]["log_loss"]).flatten()),
        np.mean(np.array(df["bmm"]["log_loss"]).flatten()),
        np.mean(np.array(df["knn"]["log_loss"]).flatten()),
        np.mean(np.array(df["baseline"]["log_loss"]).flatten()),
    ]

    # TODO: Take from actual errors
    yerrs = [
        np.std(np.array(df["log_reg"]["log_loss"]).flatten()),
        np.std(np.array(df["bmm"]["log_loss"]).flatten()),
        np.std(np.array(df["knn"]["log_loss"]).flatten()),
        np.std(np.array(df["baseline"]["log_loss"]).flatten()),
    ]
    x_labels = ["LogReg", "BMM", "KNN", "Baseline"]
    x_pos = [1, 2, 3, 4]
    ax[pnum].scatter(x_pos, ys, c="hcdarknavy")
    ax[pnum].errorbar(
        x=x_pos,
        y=ys,
        yerr=yerrs,
        markersize=0,
        linewidth=0,
        elinewidth=2,
        capsize=4,
        capthick=2,
        ecolor="hcdarknavy",
    )
    ax[pnum].set_xticks(x_pos, labels=x_labels)
    ax[pnum].set_ylabel("Log loss")


##How does each model do on test data - accuracy?

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 16), dpi=300)
ax = ax.ravel()
for pnum in range(13):
    with open(
        f"../data/outputs/fits/paper_model_fitting_results_{pnum}.json",
        encoding="utf-8",
    ) as f:
        results = json.load(f)
    df = results["AllQuestions_test_performance"]
    ys = [
        np.mean(np.array(df["log_reg"]["accuracy"]).flatten()),
        np.mean(np.array(df["bmm"]["accuracy"]).flatten()),
        np.mean(np.array(df["knn"]["accuracy"]).flatten()),
        np.mean(np.array(df["baseline"]["accuracy"]).flatten()),
    ]

    # TODO: Take from actual errors
    yerrs = [
        np.std(np.array(df["log_reg"]["accuracy"]).flatten()),
        np.std(np.array(df["bmm"]["accuracy"]).flatten()),
        np.std(np.array(df["knn"]["accuracy"]).flatten()),
        np.std(np.array(df["baseline"]["accuracy"]).flatten()),
    ]
    x_labels = ["LogReg", "BMM", "KNN", "Baseline"]
    x_pos = [1, 2, 3, 4]
    ax[pnum].scatter(x_pos, ys, c="hcdarknavy")
    ax[pnum].errorbar(
        x=x_pos,
        y=ys,
        yerr=yerrs,
        markersize=0,
        linewidth=0,
        elinewidth=2,
        capsize=4,
        capthick=2,
        ecolor="hcdarknavy",
    )
    ax[pnum].set_xticks(x_pos, labels=x_labels)
    ax[pnum].set_ylabel("Accuracy")


fig.savefig("../figs/ModelTestPerformances_accuracy_ByExam.png")


##How does each model do on test data - recall?

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 16), dpi=300)
ax = ax.ravel()
for pnum in range(13):
    with open(
        f"../data/outputs/fits/paper_model_fitting_results_{pnum}.json",
        encoding="utf-8",
    ) as f:
        results = json.load(f)
    df = results["AllQuestions_test_performance"]
    ys = [
        np.mean(np.array(df["log_reg"]["recall"]).flatten()),
        np.mean(np.array(df["bmm"]["recall"]).flatten()),
        np.mean(np.array(df["knn"]["recall"]).flatten()),
        np.mean(np.array(df["baseline"]["recall"]).flatten()),
    ]

    # TODO: Take from actual errors
    yerrs = [
        np.std(np.array(df["log_reg"]["recall"]).flatten()),
        np.std(np.array(df["bmm"]["recall"]).flatten()),
        np.std(np.array(df["knn"]["recall"]).flatten()),
        np.std(np.array(df["baseline"]["recall"]).flatten()),
    ]
    x_labels = ["LogReg", "BMM", "KNN", "Baseline"]
    x_pos = [1, 2, 3, 4]
    ax[pnum].scatter(x_pos, ys, c="hcdarknavy")
    ax[pnum].errorbar(
        x=x_pos,
        y=ys,
        yerr=yerrs,
        markersize=0,
        linewidth=0,
        elinewidth=2,
        capsize=4,
        capthick=2,
        ecolor="hcdarknavy",
    )
    ax[pnum].set_xticks(x_pos, labels=x_labels)
    ax[pnum].set_ylabel("Recall")


fig.savefig("../figs/ModelTestPerformances_recall_ByExam.png")


##How does each model do on test data - f1?

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 16), dpi=300)
ax = ax.ravel()
for pnum in range(13):
    with open(
        f"../data/outputs/fits/paper_model_fitting_results_{pnum}.json",
        encoding="utf-8",
    ) as f:
        results = json.load(f)
    df = results["AllQuestions_test_performance"]
    ys = [
        np.mean(np.array(df["log_reg"]["f1s"]).flatten()),
        np.mean(np.array(df["bmm"]["f1s"]).flatten()),
        np.mean(np.array(df["knn"]["f1s"]).flatten()),
        np.mean(np.array(df["baseline"]["f1s"]).flatten()),
    ]

    # TODO: Take from actual errors
    yerrs = [
        np.std(np.array(df["log_reg"]["f1s"]).flatten()),
        np.std(np.array(df["bmm"]["f1s"]).flatten()),
        np.std(np.array(df["knn"]["f1s"]).flatten()),
        np.std(np.array(df["baseline"]["f1s"]).flatten()),
    ]
    x_labels = ["LogReg", "BMM", "KNN", "Baseline"]
    x_pos = [1, 2, 3, 4]
    ax[pnum].scatter(x_pos, ys, c="hcdarknavy")
    ax[pnum].errorbar(
        x=x_pos,
        y=ys,
        yerr=yerrs,
        markersize=0,
        linewidth=0,
        elinewidth=2,
        capsize=4,
        capthick=2,
        ecolor="hcdarknavy",
    )
    ax[pnum].set_xticks(x_pos, labels=x_labels)
    ax[pnum].set_ylabel("F1")


fig.savefig("../figs/ModelTestPerformances_f1_ByExam.png")

##How does each model do on test data - matthews_corrcoef?

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 16), dpi=300)
ax = ax.ravel()
for pnum in range(13):
    with open(
        f"../data/outputs/fits/paper_model_fitting_results_{pnum}.json",
        encoding="utf-8",
    ) as f:
        results = json.load(f)
    df = results["AllQuestions_test_performance"]
    ys = [
        np.mean(np.array(df["log_reg"]["matthews_corrcoef"]).flatten()),
        np.mean(np.array(df["bmm"]["matthews_corrcoef"]).flatten()),
        np.mean(np.array(df["knn"]["matthews_corrcoef"]).flatten()),
        np.mean(np.array(df["baseline"]["matthews_corrcoef"]).flatten()),
    ]

    # TODO: Take from actual errors
    yerrs = [
        np.std(np.array(df["log_reg"]["matthews_corrcoef"]).flatten()),
        np.std(np.array(df["bmm"]["matthews_corrcoef"]).flatten()),
        np.std(np.array(df["knn"]["matthews_corrcoef"]).flatten()),
        np.std(np.array(df["baseline"]["matthews_corrcoef"]).flatten()),
    ]
    x_labels = ["LogReg", "BMM", "KNN", "Baseline"]
    x_pos = [1, 2, 3, 4]
    ax[pnum].scatter(x_pos, ys, c="hcdarknavy")
    ax[pnum].errorbar(
        x=x_pos,
        y=ys,
        yerr=yerrs,
        markersize=0,
        linewidth=0,
        elinewidth=2,
        capsize=4,
        capthick=2,
        ecolor="hcdarknavy",
    )
    ax[pnum].set_xticks(x_pos, labels=x_labels)
    ax[pnum].set_ylabel("Matthews Correlation Coeffience")


fig.savefig("../figs/ModelTestPerformances_matthews_corrcoef_ByExam.png")


# How does the performance aggregate in total? for each of the metrics

# How many clusters are chosen by the bmm for each exam?

fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(15, 8))
ax = ax.ravel()


n_picked = []
for pnum in range(1):
    with open(
        f"../data/outputs/fits/paper_model_fitting_results_{pnum}.json",
        encoding="utf-8",
    ) as f:
        results = json.load(f)
    df = results["AllQuestions_test_performance"]

    ax[pnum].hist(df["bmm"]["chosen_n"])
    ax[pnum].set_xlabel("Number of clusters chosen")
    ax[pnum].set_ylabel("Frequency")
    ax[pnum].set_title(f"Exam {pnum}")
    n_picked.extend(df["bmm"]["chosen_n"])

ax[-1].hist(n_picked)
ax[-1].set_ylabel("Frequency")
ax[-1].set_xlabel("Number of clusters")
ax[-1].set_title("All exams")

fig.savefig("../figs/ModelTestPerformances_BMM_N_clusters_by_Exam.png")


# What do the cluster profiles look like?

with open(
    f"../data/outputs/fits/paper_model_fitting_results_{pnum}.json",
    encoding="utf-8",
) as f:
    results = json.load(f)
df = results["AllQuestions_test_performance"]

cluster_coords = df["bmm"]["cluster_coords"][0]  # need to choose a q number, 0

n_cls = len(cluster_coords)
fig, ax = plt.subplots(
    nrows=ceil(n_cls / 4.0),
    ncols=4,
    figsize=(n_cls / 4, 8),
    dpi=300,
    sharex=True,
    sharey=True,
)

ax = ax.ravel()


for mu in range(n_cls):
    ax[mu].plot(cluster_coords[mu])
    if mu % 4 == 0:
        ax[mu].set_ylabel("Probability")
    if mu > n_cls - 4:
        ax[mu].set_xlabel("Question number")

fig.savefig("../figs/ModelTestPerformances_Example_BMM_cluster_coords.png")


# How does test performance compare to validation performance?
