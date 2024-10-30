from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class BaseModel(ABC):
    """
    Base class for machine learning models implemented from scratch, using gradient descent optimization.

    Attributes:
    - alpha (float): Learning rate for gradient descent.
    - iterations (int): Number of iterations for gradient descent.
    - random_init_params (bool): If True, initialize parameters randomly.
    - verbose (int): Verbosity level for gradient descent progress (0: silent, 1: detailed).
    """

    def __init__(
        self,
        alpha: float = 0.001,
        iterations: int = 500,
        random_init_params: bool = False,
        verbose: int = 0,
    ) -> None:
        """
        Initializes the base model with gradient descent hyperparameters.

        Parameters:
        - alpha (float): Learning rate for gradient descent.
        - iterations (int): Number of iterations for gradient descent.
        - random_init_params (bool): Whether to randomly initialize model parameters.
        - verbose (int): Verbosity level (0: silent, 1: verbose).
        """
        self.alpha: float = alpha
        self.iterations: int = iterations
        self.random_init_params: bool = random_init_params
        self.verbose: int = verbose
        self.theta: Optional[np.ndarray] = None  # Parameters will be initialized in child classes

    @abstractmethod
    def model_name(self) -> str:
        """
        Indicates the name of the model
        """
        pass

    @abstractmethod
    def hypothesis_function(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the hypothesis (predictions) for the given feature matrix X.

        Parameters:
        - X (np.ndarray): Feature matrix.

        Returns:
        - np.ndarray: Predicted values.
        """
        pass

    @abstractmethod
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the cost for the current state of the model.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target values.

        Returns:
        - float: The computed cost.
        """
        pass

    @abstractmethod
    def gradient_descent(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Performs gradient descent to minimize the cost function and optimizes parameters.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target values.

        Returns:
        - Tuple[np.ndarray, list]: Optimized parameters and the history of cost values.
        """
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the model using the training data and stores the optimized parameters.

        Parameters:
        - X_train (np.ndarray): Training feature matrix.
        - y_train (np.ndarray): Training target values.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.theta, self.cost_history = self.gradient_descent(X_train, y_train)
        if self.verbose:
            print("Training complete. Optimal parameters stored.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generates predictions on new data using the learned parameters.

        Parameters:
        - X_test (np.ndarray): Feature matrix for the test set.

        Returns:
        - np.ndarray: Predicted target values for the test set.
        """
        if self.theta is None:
            raise ValueError("Model parameters not initialized. Please run fit() first.")
        return self.hypothesis_function(X_test)
