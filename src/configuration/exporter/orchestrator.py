import os
import pickle
import pandas as pd
from typing import Optional, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq
from configuration.model_algo.basemodel import BaseModel


class Exporter:
    def __init__(
        self,
        model: BaseModel,
        output_path: str,
        pipeline: Optional[Any] = None,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        y_test: Optional[pd.Series] = None,
    ):
        """
        Initializes the Exporter with the model, pipeline, training and testing data, and optional log path.

        Parameters:
        - model (Any): The trained model to be exported.
        - pipeline (Optional[Any]): The data processing pipeline used for the model.
        - X_train (Optional[pd.DataFrame]): Training feature set.
        - X_test (Optional[pd.DataFrame]): Testing feature set.
        - y_train (Optional[pd.Series]): Training target set.
        - y_test (Optional[pd.Series]): Testing target set.
        - log_path (Optional[str]): Path to the log file for export.
        - output_path (str): output path where to store output folder.
        """
        self.model = model
        self.pipeline = pipeline
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if output_path is None:
            raise ValueError("No Ouput path specified !")
        else:
            self.experiment_dir = output_path

        os.makedirs(self.experiment_dir, exist_ok=True)

    def export_model(self) -> str:
        """
        Saves the model to a pickle file in the experiment directory.

        Returns:
        - str: Path to the saved model file.
        """
        model_path = os.path.join(self.experiment_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        return model_path

    def export_pipeline(self) -> Optional[str]:
        """
        Saves the data processing pipeline to a pickle file if it exists.

        Returns:
        - Optional[str]: Path to the saved pipeline file, or None if pipeline is not provided.
        """
        if self.pipeline:
            pipeline_path = os.path.join(self.experiment_dir, "pipeline.pkl")
            with open(pipeline_path, "wb") as f:
                pickle.dump(self.pipeline, f)
            return pipeline_path
        else:
            print("No pipeline to export.")
            return None

    def export_data(self) -> Dict[str, str]:
        """
        Saves the training and testing datasets in parquet format in the experiment directory.

        Returns:
        - Dict[str, str]: Paths to the saved training and testing data files.
        """
        data_paths = {}

        if self.X_train is not None and self.y_train is not None:
            train_data = pd.concat([self.X_train, self.y_train.rename("target")], axis=1)
            train_data_path = os.path.join(self.experiment_dir, "train_data.parquet")
            train_table = pa.Table.from_pandas(train_data)
            pq.write_table(train_table, train_data_path)
            data_paths["train_data"] = train_data_path

        if self.X_test is not None and self.y_test is not None:
            test_data = pd.concat([self.X_test, self.y_test.rename("target")], axis=1)
            test_data_path = os.path.join(self.experiment_dir, "test_data.parquet")
            test_table = pa.Table.from_pandas(test_data)
            pq.write_table(test_table, test_data_path)
            data_paths["test_data"] = test_data_path

        return data_paths

    def export_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Saves the model performance metrics to a CSV file in the experiment directory.

        Parameters:
        - metrics (Dict[str, float]): Dictionary containing the performance metrics of the model.

        Returns:
        - str: Path to the saved metrics file.
        """
        metrics_path = os.path.join(self.experiment_dir, "model_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        return metrics_path
