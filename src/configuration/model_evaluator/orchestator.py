import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from configuration.model_algo.basemodel import BaseModel


class ModelEvaluator:
    def __init__(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray):
        """
        Initialize the evaluator with model, test features, and true labels.
        """
        self.model: BaseModel = model
        self.X_test: np.ndarray = X_test
        self.y_test: np.ndarray = y_test

    def evaluate_regression(self):
        """
        Evaluates the model and returns a dictionary of performance metrics.
        """
        predictions = self.model.predict(self.X_test)
        metrics = {
            "Model": self.model.model_name(),
            "MSE": mean_squared_error(self.y_test, predictions),
            "MAE": mean_absolute_error(self.y_test, predictions),
            "R2_Score": r2_score(self.y_test, predictions),
        }
        return metrics
