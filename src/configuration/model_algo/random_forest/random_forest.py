from typing import Tuple, List, Optional
import numpy as np
from configuration.model_algo.basemodel import BaseModel
from sklearn.tree import DecisionTreeRegressor


class RandomForest(BaseModel):
    """
    Implements Random Forest from scratch with ensemble learning using decision trees.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Optional[int] = None,
        bootstrap: bool = True,
        max_depth: Optional[int] = None,
        min_samples_split: float = 0.1,
        verbose: int = 0,
    ) -> None:
        """
        Initializes the Random Forest model.

        Parameters:
        - n_estimators (int): Number of trees in the forest.
        - max_features (Optional[int]): Maximum number of features to consider for splitting a node.
        - bootstrap (bool): If True, each tree is trained on a bootstrap sample.
        - max_depth (Optional[int]): Maximum depth of the trees.
        - min_samples_split (int): Minimum number of samples required to split an internal node.
        - verbose (int): Verbosity level (0: silent, 1: detailed).
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.verbose = verbose
        self.trees: List[DecisionTreeRegressor] = []

    def model_name(self) -> str:
        """
        Indicates the name of the model.

        Returns:
        - model_name (str): Name of the model.
        """
        return "random_forest"

    def hypothesis_function(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the ensemble predictions by averaging the predictions of each tree.

        Parameters:
        - X (np.ndarray): Feature matrix.

        Returns:
        - np.ndarray: Predicted values.
        """
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the Mean Squared Error (MSE) for the Random Forest model.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target values.

        Returns:
        - float: The computed MSE.
        """
        predictions = self.hypothesis_function(X)
        mse = np.mean((y - predictions) ** 2)
        return mse

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the Random Forest model using training data.

        Parameters:
        - X (np.ndarray): Training feature matrix.
        - y (np.ndarray): Training target values.
        """
        y = y.values.reshape(-1, 1)
        n_samples, n_features = X.shape
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for i in range(self.n_estimators):
            if self.verbose:
                print(f"Training tree {i + 1}/{self.n_estimators}")

            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample, y_sample = X, y

            # Create and fit a decision tree regressor
            tree = DecisionTreeRegressor(
                max_features=self.max_features, max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def optimization_method(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Placeholder method. optimization_method is not used in Random Forest.
        """
        raise NotImplementedError("optimization_method is not applicable to Random Forest.")
