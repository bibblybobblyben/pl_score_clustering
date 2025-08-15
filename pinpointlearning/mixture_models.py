from abc import ABC
from typing import List, Tuple  # noqa

import matplotlib.pyplot as plt  # type: ignore
import mlflow
import numpy as np

from pinpointlearning.utils import (
    calculate_scores,
    match_vectors,
    calculate_bernoulli_prob,
)


class MixtureModel(ABC):
    def __init__(self):
        """Base class for mixture models. Minimally required classes defined"""
        self.aic = -np.inf
        self.bic = -np.inf
        self.ilc = -np.inf
        self.log_likelihood = -np.inf

    def fit(self, X) -> None:
        return None

    def evaluate(self, vecs) -> Tuple:
        return ()


class BernoulliMixture(MixtureModel):
    def __init__(
        self, n_components, tol, max_iter, alpha_mu=0.01, alpha_pi=0.01, use_mlflow=True
    ) -> None:
        """Estimate parameters of mixture of Bernoulli distributions using expectation
        maximisation approach.

        Args:
            n_components (int): Number of mixture components.
            tol (float): Convergence threshold.
            max_iter (int): Maximum number of iterations.
            mu_alpha (float): Alpha smoothing parameter when updating mu estimates.
            pi_alpha (float): Alpha smoothing parameter when updating pi estimates.
        """

        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.alpha_mu = alpha_mu
        self.alpha_pi = alpha_pi
        if use_mlflow:
            mlflow.log_param("n_components", n_components)
        self.use_mlflow = use_mlflow

        self.n_iter = 0
        self.cluster_lks = np.array([])
        self.log_likelihood = 0
        self.q = np.array([])
        self.n_k = None
        self.mu = np.array([])
        self.pi = None
        self.aic = None
        self.bic = None

        self.cluster_lk_values = []  # type: List[List]
        self.q_values = []  # type: List[np.typing.NDArray]
        self.n_k_values = []  # type: List[np.typing.NDArray]
        self.mu_values = []  # type: List[np.typing.NDArray]
        self.pi_values = []  # type: List[List]

        self.log_likelihood_values = []  # type: List[float]

    def _initialise(self, X) -> None:
        """Set initial values for q, mu and pi.

        q (n_students, n_clusters) are the estimated probabilties of each observation
        belonging to each cluster. Initial values are set to reciprocal of the number of
        components.

        mu (n_clusters, n_questions) are the estimated Bernoulli distribution parameters
        for each cluster. Initial values are set to random deviates from the uniform
        distribution [0,1).

        pi (n_clusters) are the estimated probabilities of belonging to each cluster.
        Initial values are set to the reciprocal of the number of components.

        Args:
            X (ndarray[n_students, n_questions]): Data.
        """

        n_students, n_questions = X.shape
        self.expected_shape = X.shape

        self.q = np.ones([n_students, self.n_components]) / self.n_components
        self.mu = np.random.rand(self.n_components, n_questions)
        self.pi = np.ones(self.n_components) / self.n_components

    def _bernoulli_prob(self, mu, X_s):

        return calculate_bernoulli_prob(mu, X_s)

    def _assignment_probs(self, bernoulli_pr):
        return self.pi * np.prod(bernoulli_pr, axis=2)

    def _update_cluster_lks_value(self, X, store_probs=True) -> None:
        """Calculate cluster likelihood values using current parameter
        estimates.

        Args:
            X (ndarray[n_students, n_questions]): Data.
        """

        X_s = np.stack([X] * self.n_components, axis=1)

        bernoulli_pr = self._bernoulli_prob(mu=self.mu, X_s=X_s)
        clusters_lks = self._assignment_probs(bernoulli_pr=bernoulli_pr)

        self.cluster_lks = clusters_lks
        self.cluster_lk_values.append(clusters_lks)

    def _update_q_value(self) -> None:
        """Update q (n_students, n_clusters) estimates i.e., the probabilities of each
        observation belonging to each cluster.
        """

        q = self.cluster_lks.T / self.cluster_lks.T.sum(axis=0)

        self.q = q
        self.q_values.append(q)

    def _update_n_k_value(self) -> None:
        """Update N_k (n_clusters) estimate, i.e, "the effective number of data point
        associated with component k" (Bishop and Nasrabadi, 2006, p.446)
        """

        n_k = self.q.sum(axis=1)

        self.n_k = n_k
        self.n_k_values.append(n_k)

    def _update_mu_value(self, X) -> None:
        """Update mu (n_clusters, n_questions) estimate, i.e, Bernoulli distribution
        parameters for each cluster.

        Args:
            X (ndarray[n_students, n_questions]): Data.
        """

        alpha = self.alpha_mu
        n_questions = X.shape[1]
        mu = ((self.q.dot(X).T + alpha) / (self.n_k + alpha * n_questions)).T

        self.mu = mu
        self.mu_values.append(mu)

    def _update_pi_value(self, X) -> None:
        """Update pi (n_clusters) estimates, i.e., probability of component k

        Args:
            X (ndarray[n_students, n_questions]): Data.
        """

        pi = self.alpha_pi
        n_students = X.shape[0]
        pi = (self.n_k + pi) / (n_students + pi * self.n_components)

        self.pi = pi
        self.pi_values.append(pi)

    def _e_step(self, X) -> None:
        """Expectation step to update q estimates.

        Args:
            X (ndarray[n_students, n_questions]): Data.
        """

        self._update_cluster_lks_value(X)
        self._update_q_value()

    def _m_step(self, X) -> None:
        """Maximisation step to update N_k, mu and pi estimates.

        Args:
            X (ndarray[n_students, n_questions]): Data.
        """

        self._update_n_k_value()
        self._update_mu_value(X)
        self._update_pi_value(X)

    def _calculate_log_likelihood(self) -> None:
        """Calculate log likelihood given current parameter estimates."""

        sum_cluster_lks = self.cluster_lks.sum(axis=1)
        self.log_likelihood = np.ma.log(sum_cluster_lks).sum()
        self.log_likelihood_values.append(self.log_likelihood)
        if self.use_mlflow:
            mlflow.log_metric("log_likelihood", self.log_likelihood, step=self.n_iter)

    def _check_convergence(self) -> bool:
        """Check if log likelihood has converged to within given tolerance.

        Returns:
            bool: Converged.
        """

        diff = self.log_likelihood_values[-2] - self.log_likelihood_values[-1]
        return bool(abs(diff) < self.tol)

    def _calculate_aic(self, X) -> None:
        """Calculate Akaike information criterion.

        Args:
            X (ndarray[n_students, n_questions]): Data.
        """

        n_questions = X.shape[1]
        n_parameters = self.n_components * (n_questions + 1) - 1
        self.aic = 2 * n_parameters - 2 * self.log_likelihood
        if self.use_mlflow:
            mlflow.log_metric("aic", self.aic)

    def _calculate_bic(self, X) -> None:
        """Calculate Bayesian information criterion.

        Args:
            X (ndarray[n_students, n_questions]): Data.
        """

        n_students, n_questions = X.shape
        n_parameters = self.n_components * (n_questions + 1) - 1
        self.bic = n_parameters * np.log(n_students) - 2 * self.log_likelihood
        if self.use_mlflow:
            mlflow.log_metric("bic", self.bic)

    def _calculate_ilc(self, X) -> None:
        self._calculate_bic(X)
        n_parameters = self.n_components * (X.shape[1] + 1) - 1
        self.ilc = self.bic - 0.5 * n_parameters * np.log(X.shape[0])
        if self.use_mlflow:
            mlflow.log_metric("ilc", self.ilc)

    def fit(self, X) -> None:
        """Estimate parameters of mixture of Bernoulli distributions using expectation
        maximisation approach.

        Args:
            X (ndarray[n_students, n_questions]): Data.
        """

        self._initialise(X)
        self.n_iter = 0

        while self.n_iter < self.max_iter:

            if self.n_iter > 2:
                if self._check_convergence():
                    break

            self._e_step(X)
            self._m_step(X)
            self._calculate_log_likelihood()

            self.n_iter += 1

        self._calculate_aic(X)
        self._calculate_bic(X)
        self._calculate_ilc(X)

    def evaluate(self, vecs) -> Tuple:
        """Given a list of vectors, calculate mapping from vectors found by EM
        algorithm, scores for each mapping, and sum of scores for mappings. Note, the
        scoring is only for matched vectors, not all vectors will be matched if the list
        of vectors are of different lengths.

        Args:
            vecs (ndarray): List of vectors.

        Returns:
            [dict, [float], float]: Mapping from vectors found by EM algorithm,
            scores for each mapping, and sum of scores for mappings.
        """

        vecs_1 = self.mu
        vecs_2 = vecs

        vecs_map = match_vectors(vecs_1, vecs_2)
        scores = calculate_scores(vecs_1, vecs_2, vecs_map)

        n_matchs = scores.shape[0]

        sum_matched_scores = scores.sum()
        if self.use_mlflow:
            mlflow.log_metric("sum_matched_scores", sum_matched_scores)

        mean_matched_score = sum_matched_scores / n_matchs
        if self.use_mlflow:
            mlflow.log_metric("mean_matched_score", mean_matched_score)

        return vecs_map, scores, sum_matched_scores, mean_matched_score

    def plot_log_likelihood_values(self) -> plt.plot:
        """Plot log likelihodd against iteration number.

        Returns:
            px.line: Plot.
        """

        plt.style.use("seaborn-v0_8-whitegrid")

        plt.plot(self.log_likelihood_values)
        plt.xlabel("Iteration")
        plt.ylabel("Log Likelihood")
        plt.show()

    def predict(self, X, pred_mask):
        """Given some new observations,
        perform probabilistic assignments to each class

        Args:
            target (_type_): Observations for assignment
            col_predict (int): column index of the data to
            assign a prediction for
        """

        if X.shape != self.expected_shape:
            raise ValueError(
                "Array for prediction needs to be the same shape as "
                f"training data. Expected {(self.expected_shape)} but got {X.shape}"
            )

        # TODO: how do we allow masking of columns?
        print("mask", pred_mask)

        X_s = np.stack([X[:, ~pred_mask]] * self.n_components, axis=1)

        mu = self.mu[:, ~pred_mask]
        pi = np.array(self.pi)

        bernoulli_pr = np.prod(self._bernoulli_prob(mu=mu, X_s=X_s), axis=2)

        normed_assignment_probs = np.zeros((X.shape[0], len(pi)))
        raw_assignment_probs = np.zeros(normed_assignment_probs.shape)

        # get the probability of each row belonging to each cluster
        for group in range(self.n_components):
            print("a", mu[group].shape)
            print("b", self.cluster_lks[:, group].shape)
            # TODO: make robust to multi column
            # print("assprob", clusters_lks[:,group][:5])
            group_assignment_prob = (
                pi[group] * np.array(bernoulli_pr[:, group])
            ).reshape(-1)

            raw_assignment_probs[:, group] = group_assignment_prob

        #        totprobs = np.sum(raw_assignment_probs, axis =1)

        normed_assignment_probs = raw_assignment_probs / np.sum(
            raw_assignment_probs, axis=1
        ).reshape((-1, 1))

        # calculate the weighted sum of probabilities
        predictions = np.zeros(X.shape[0])

        for i in range(self.n_components):
            weighted_pred = normed_assignment_probs[:, i] * self.mu[i, pred_mask]
        return weighted_pred
