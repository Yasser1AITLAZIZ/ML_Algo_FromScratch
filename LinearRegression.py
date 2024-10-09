import numpy as np
from sklearn.model_selection import train_test_split


class LinearRegression:
    """
    Implements Linear Regression from scratch with gradient descent optimization.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.001,
        iterations: int = 500,
        interception: bool = False,
        random_init_params: bool = False,
        verbose: int = 0,
    ):
        """
        Initializes the model with the input data and hyperparameters.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target variable.
        - alpha (float): Learning rate for gradient descent.
        - iterations (int): Number of iterations for gradient descent.
        - interception (bool): Whether to add an intercept term.
        - random_init_params (bool): Whether to randomly initialize the parameters.
        - verbose (int): Level of verbosity for displaying cost during training (0: silent, 1: verbose).
        """
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.interception: bool = interception
        self.random_init_params: bool = random_init_params
        self.alpha: float = alpha
        self.iterations: int = iterations
        self.verbose: int = verbose

    def split_train_test(self) -> tuple:
        """
        Splits the dataset into training and test sets and adds an intercept term if required.

        Returns:
        - tuple: Split of X_train, X_test, y_train, y_test.
        """
        if self.interception:
            # Add a column of ones to include the intercept term
            self.X = np.c_[np.ones(self.X.shape[0]), self.X]

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def initialization_function(self) -> None:
        """
        Initializes the parameters (theta) for gradient descent either with zeros or randomly.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()

        if self.random_init_params:
            self.theta = np.random.rand(self.X_train.shape[1])
        else:
            self.theta = np.zeros(self.X_train.shape[1])

    def hypothesis_function(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the hypothesis (predictions) for the given feature matrix X.

        Parameters:
        - X (np.ndarray): Feature matrix.

        Returns:
        - np.ndarray: Predictions.
        """
        return np.dot(X, self.theta)

    def compute_cost(self) -> float:
        """
        Computes the cost (Mean Squared Error) for the current state of the model.

        Returns:
        - float: The computed cost.
        """
        predictions = self.hypothesis_function(self.X_train)
        errors = predictions - self.y_train
        cost = (1 / (2 * len(self.y_train))) * np.sum(errors**2)
        return cost

    def gradient_descent(self) -> tuple:
        """
        Performs gradient descent to minimize the cost function.

        Returns:
        - tuple: The optimized parameters (theta) and the history of the cost during training.
        """
        self.initialization_function()
        self.cost_history = []

        for i in range(self.iterations):
            # Compute the cost and optionally print it
            cost = self.compute_cost()
            self.cost_history.append(cost)
            if self.verbose == 1:
                print(f"Iteration {i+1}: Cost = {cost}")

            # Calculate predictions and errors
            predictions = self.hypothesis_function(self.X_train)
            errors = predictions - self.y_train

            # Update parameters (theta)
            gradient = (1 / len(self.y_train)) * np.dot(self.X_train.T, errors)
            self.theta = self.theta - self.alpha * gradient

        return self.theta, self.cost_history

    def train(self) -> None:
        self.theta_optim, cost_hist = self.gradient_descent()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data using the learned parameters.

        Parameters:
        - X (np.ndarray): Feature matrix for the test set.

        Returns:
        - np.ndarray: Predicted target values.
        """
        return self.hypothesis_function(X)

    def metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        Computes the Metrics between true and predicted values.

        Parameters:
        - y_true (np.ndarray): True target values.
        - y_pred (np.ndarray): Predicted target values.

        Returns:
        - float: The computed MSE & MAE.
        """
        mse = (1 / len(y_true)) * np.sum((y_pred - y_true) ** 2)
        mae = (1 / len(y_true)) * np.sum((y_pred - y_true))
        return mse, mae

    def evaluate_test(self) -> None:
        """
        Evaluates the model on the test set and prints the Mean Squared Error (MSE).
        """
        y_pred = self.predict(self.X_test)
        mse_test, mae_test = self.metrics(self.y_test, y_pred)
        print(f"MSE/MAE on the test set: {mse_test}/{mae_test}")
