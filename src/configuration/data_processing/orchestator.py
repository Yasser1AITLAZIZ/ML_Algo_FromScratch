import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataProcessing:
    def __init__(self, config: dict):
        """
        Initializes the data processing pipeline based on the configuration.

        Parameters:
        - config (dict): Configuration dictionary with data processing settings.
        """
        self.config = config

    def load_data(self) -> tuple:
        """
        Loads dataset and splits into features and target based on config.

        Returns:
        - tuple: DataFrame X (features), Series y (target)
        """
        data = pd.read_csv(self.config["data_path"])
        X = data.drop(columns=[self.config["target_column"]])
        y = data[self.config["target_column"]]
        return X, y

    def setup_pipeline(self, X: pd.DataFrame):
        """
        Configures the data transformation pipeline based on feature types.

        Parameters:
        - X (pd.DataFrame): Input features for the pipeline configuration.
        """
        # Define numerical and categorical features
        numeric_features = (
            X.columns.tolist()
            if self.config.get("numeric_features") == "all"
            else self.config.get("numeric_features", X.select_dtypes(include=["float64", "int64"]).columns.tolist())
        )
        categorical_features = (
            X.columns.tolist()
            if self.config.get("categorical_features") == "all"
            else self.config.get("categorical_features")
        )

        # Set up numerical transformer based on config
        if self.config["scaling"] == "standard":
            numeric_transformer = StandardScaler()
        elif self.config["scaling"] == "minmax":
            numeric_transformer = MinMaxScaler()
        else:
            numeric_transformer = "passthrough"  # No scaling

        # Set up categorical transformer if required
        if self.config.get("one_hot_encode", False):
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        else:
            categorical_transformer = "passthrough"

        # Combine transformations
        transformers = []
        if numeric_features is not None:
            transformers.append(("num", numeric_transformer, numeric_features))
        if categorical_features is not None:
            transformers.append(("cat", categorical_transformer, categorical_features))

        self.preprocessor = ColumnTransformer(
            transformers=transformers, remainder="passthrough"  # Keeps any columns not specified
        )

    def transform_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        Fits the preprocessing pipeline on the training data and transforms both train and test data.

        Parameters:
        - X_train (pd.DataFrame): Training features
        - X_test (pd.DataFrame): Test features

        Returns:
        - tuple: Transformed X_train and X_test as numpy arrays
        """
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        return X_train_transformed, X_test_transformed
