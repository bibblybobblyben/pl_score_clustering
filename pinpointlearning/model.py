"""
Functions for modelling exam results
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class Model(ABC):
    """
    ABC for a model enacting the required functionality in this work.
    Streamlines comparison of models downstream when we can ensure all models
    will enable the same in/out interaction
    """

    def __init__(self) -> None:
        self.data = np.array([])
        self.target = np.array([])
        self.sf = False

    def preprocess(self, features: np.array, target: np.array | None = None):
        """Preprocess data such that it is in a consistent form for modelling.

        Args:
            features (np.array): Array of n_students x n_questions, containing
            scores on each question
            target (np.array | None, optional): Target variable. Only used
            in training. Defaults to None.
        """
        self.data = np.divide(features, np.amax(features, axis=1).reshape(-1, 1))
        self.data = np.nan_to_num(self.data)
        if self.sf:
            self.data = np.mean(self.data, axis=1).reshape(-1, 1)
        if target is not None:
            self.target = target.reshape(-1)

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
    on the mean score across all questions and ive made an edit
    """

    def __init__(self, single_feature=False) -> None:
        super().__init__()
        self.lr = LogisticRegression()
        self.sf = single_feature

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
        self.preprocess(features=features)
        return self.lr.predict_proba(self.data)


class KNN(Model):
    """Applies a K nearest neighbours algorithm to data"""

    def __init__(self, n_neighbours=10) -> None:
        super().__init__()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbours)

    def fit(self, features: np.array, target: np.array):
        """Train a KNearestNeighbours model according to inputs

        Args:
            features (np.array): coordinates for KNN clustering
            target (np.array): labels for each row in features
        """
        self.preprocess(features=features, target=target)
        self.knn.fit(X=self.data, y=self.target)

    def predict(self, features: np.array):
        """label new samples using the KNN stored on the class

        Args:
            features (np.array): samples to identify

        Returns:
            np.array: labels for each sample
        """
        self.preprocess(features=features)
        return self.knn.predict(X=self.data)

    def predict_proba(self, features: np.array):
        """Probabilities of each sample being in each of the target classes

        Args:
            features (np.array): Coordinates to classify with KNN

        Returns:
            np.array: Probabilities of being in each target class
        """
        self.preprocess(features=features)
        return self.knn.predict_proba(X=self.data)
