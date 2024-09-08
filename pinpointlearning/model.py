"""
Functions for modelling exam results
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression


class Model(ABC):
    """
    ABC for a model enacting the required functionality in this work.
    Streamlines comparison of models downstream when we can ensure all models
    will enable the same in/out interaction
    """

    def __init__(self) -> None:
        return

    @abstractmethod
    def fit(self, features: np.array, target: np.array):
        """
        Train the model on new data `features`, and target `target`
        """

    @abstractmethod
    def predict(self, features: np.array):
        """Predict on new data `features`

        Args:
            features (np.array): New data
        """

    @abstractmethod
    def predict_proba(self, features: np.array):
        """
        Returns probabilities for output classes for new data in `features`
        """


class LogReg(Model):
    """
    Applies a single feature logistic regression model to input data, based
    on the mean score across all questions
    """

    def __init__(self) -> None:
        super().__init__()
        self.lr = LogisticRegression()
        self.data = np.array([])
        self.target = np.array([])

    def preprocess(self, features: np.array, target: np.array | None = None):
        """Preprocess data such that it is in a consistent form for modelling.

        Args:
            features (np.array): Array of n_students x n_questions, containing
            scores on each question
            target (np.array | None, optional): Target variable. Only used
            in training. Defaults to None.
        """
        self.data = np.divide(features, np.amax(features, axis=1).reshape(-1, 1))
        self.data = np.mean(self.data, axis=1).reshape(-1, 1)
        if target is not None:
            self.target = target.reshape(-1)

    def fit(self, features: np.array, target: np.array):
        """Train the model stored on the class

        Args:
            features (np.array): Array of n_students x n_questions, containing
            scores on each question
            target (np.array): Target labels to assign to each sample in
            `features`

        """
        self.preprocess(features=features, target=target)
        self.lr.fit(self.data, self.target)

    def predict(self, features: np.array):
        """Use the model stored on the class to predict on new samples.

        Args:
            features (np.array): data points to be classified

        Returns:
            np.array: Binary labels of samples in `features`
        """
        self.preprocess(features)
        return self.lr.predict(self.data)

    def predict_proba(self, features: np.array):
        """Return the probabilities of each sample being in each class.

        Args:
            features (np.array): Data points to be classified

        Returns:
            np.array: Array of n_samples, n_classes with probabilities of each
            row being within each of the target classes [0,1]
        """
        self.preprocess(features)
        return self.lr.predict_proba(self.data)
