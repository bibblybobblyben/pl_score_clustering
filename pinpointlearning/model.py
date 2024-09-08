"""
Functions for modelling exam results
"""
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """
    ABC for a model enacting the required functionality in this work.
    Streamlines comparison of models downstream when we can ensure all models
    will enable the same in/out interaction
    """

    def __init__(self) -> None:
        return

    @abstractmethod
    def train(self, features: np.array, target: np.array):
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
