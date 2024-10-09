from typing import Tuple
import numpy as np
from configuration.model_algo.basemodel import BaseModel


class LinearRegression(BaseModel):
    """
    Implements Linear Regression with optional regularization (None, Lasso, Ridge, ElasticNet)
    from scratch using gradient descent optimization.
    """

    def __init__(
        self,
        alpha: float = 0.001,
        iterations: int = 500,
        random_init_params: bool = False,
        verbose: int = 0,
        intercept: bool = False,
        regularization: str = "None",  # Type of regularization: 'None', 'lasso', 'ridge', or 'elastic_net'
        lambda_: float = 0.01,  # Regularization strength
        l1_ratio: float = 0.5,  # Ratio of L1/L2 for ElasticNet, only relevant if regularization='elastic_net'
    ) -> None:
        """
        Initializes the linear regression model.

        Parameters:
        - alpha (float): Learning rate for gradient descent.
        - iterations (int): Number of iterations for gradient descent.
        - random_init_params (bool): If True, initializes model parameters randomly.
        - verbose (int): Verbosity level (0: silent, 1: detailed).
        - intercept (bool): If True, adds an intercept term.
        - regularization (str): Type of regularization to apply ('None', 'lasso', 'ridge', 'elastic_net').
        - lambda_ (float): Regularization strength.
        - l1_ratio (float): L1/L2 ratio for ElasticNet (0 <= l1_ratio <= 1).

        Raises:
        - ValueError: If an unsupported regularization type is provided or l1_ratio is out of bounds.
        """
        super().__init__(alpha, iterations, random_init_params, verbose)
        if regularization not in ["None", "lasso", "ridge", "elastic_net"]:
            raise ValueError("Unsupported regularization type. Choose from 'None', 'lasso', 'ridge', or 'elastic_net'.")

        if not (0 <= l1_ratio <= 1):
            raise ValueError("l1_ratio must be between 0 and 1 for ElasticNet regularization.")

        self.intercept = intercept
        self.regularization = regularization
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio
        self.theta: np.ndarray

    def model_name(self) -> str:
        """
        Indicates the name of the model

        Returns:
        - model_name (str): name of the model with its regularization type if used.
        """
        model_name = "linear_regression"
        if self.regularization in ["lasso", "ridge", "elastic_net"]:
            model_name += f"_{self.regularization}"

        return model_name

    def hypothesis_function(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the hypothesis (predictions) for the given feature matrix X.

        Parameters:
        - X (np.ndarray): Feature matrix.

        Returns:
        - np.ndarray: Predicted values.

        Raises:
        - ValueError: If model parameters (theta) are not initialized.
        """
        if self.theta is None:
            raise ValueError("Model parameters (theta) not initialized. Please run fit() first.")

        return np.dot(X, self.theta)

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the cost for the current state of the model, including optional regularization.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target values.

        Returns:
        - float: The computed cost.
        """

        m = len(y)
        predictions = self.hypothesis_function(X)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors**2)

        # Apply regularization if specified
        if self.regularization == "lasso":
            cost += self.lambda_ * np.sum(np.abs(self.theta[self.theta_index :]))
        elif self.regularization == "ridge":
            cost += (self.lambda_ / 2) * np.sum(self.theta[self.theta_index :] ** 2)
        elif self.regularization == "elastic_net":
            l1_penalty = self.l1_ratio * np.sum(np.abs(self.theta[self.theta_index :]))
            l2_penalty = (1 - self.l1_ratio) * np.sum(self.theta[self.theta_index :] ** 2)
            cost += self.lambda_ * (l1_penalty + l2_penalty)

        return cost

    def gradient_descent(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Performs gradient descent to minimize the cost function and optimize parameters.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target values.

        Returns:
        - Tuple[np.ndarray, list]: Optimized parameters and the history of cost values.

        Raises:
        - ValueError: If shapes of X and y are inconsistent.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")

        self.theta_index = 0
        if self.intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            self.theta_index = 1  # Exclude intercept term if exists

        m, n = X.shape
        self.theta = np.random.rand(n) if self.random_init_params else np.zeros(n)
        cost_history = []

        for i in range(self.iterations):
            predictions = self.hypothesis_function(X)
            errors = predictions - y
            gradient = (1 / m) * np.dot(X.T, errors)

            # Apply regularization to gradient if specified
            if self.regularization == "lasso":
                gradient[self.theta_index :] += self.lambda_ * np.sign(self.theta[self.theta_index :])
            elif self.regularization == "ridge":
                gradient[self.theta_index :] += self.lambda_ * self.theta[self.theta_index :]
            elif self.regularization == "elastic_net":
                l1_term = self.l1_ratio * np.sign(self.theta[self.theta_index :])
                l2_term = (1 - self.l1_ratio) * self.theta[self.theta_index :]
                gradient[self.theta_index :] += self.lambda_ * (l1_term + l2_term)

            self.theta -= self.alpha * gradient

            # Record cost for each iteration
            cost = self.compute_cost(X, y)
            cost_history.append(cost)
            if self.verbose == 1:
                print(f"Iteration {i + 1}: Cost = {cost}")

        return self.theta, cost_history
