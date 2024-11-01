import os
from datetime import datetime
from sklearn.model_selection import train_test_split

from utils.loading_config import load_config
from utils.logger import setup_logger
from configuration.model_algo.linear_regression.linear_regression import LinearRegression
from configuration.data_processing.orchestator import DataProcessing
from configuration.model_evaluator.orchestator import ModelEvaluator
from configuration.exporter.orchestrator import Exporter
from configuration.environment_variables import OUTPUT_FOLDER_PATH

# Step 1: Setup logger and record start time
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
log_path = f"{OUTPUT_FOLDER_PATH}" + "/log_file.log"
logger = setup_logger("ModelTrainingLogger", log_path)
start_time = datetime.now()
logger.info("Script execution started.")

# Step 2: Load configuration
logger.info("Loading configuration")
config = load_config("./configuration/config_main.yaml")

# Step 3: Initialize data processing pipeline.
logger.info("Initializing data processing pipeline")
data_processor = DataProcessing(config)

# Step 4: Load dataset
logger.info("Loading dataset")
X, y = data_processor.load_data()

# Step 5: Split train/test sets
logger.info("Splitting train and test sets")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"], random_state=config["random_state"]
)

# Step 6: Setup pipeline and transform data
logger.info("Setting up data processing pipeline and transforming data")
data_processor.setup_pipeline(X_train)
X_train_transformed, X_test_transformed = data_processor.transform_data(X_train, X_test)

# Step 7: Initialize models
logger.info("Initializing models")
perfs = []

for model_name in config["models"]:
    if model_name == "linear_regression":
        logger.info("Initializing Linear Regression model")
        model = LinearRegression(
            alpha=config["alpha"],
            iterations=config["iterations"],
            random_init_params=config["random_init_params"],
            verbose=config["verbose"],
            intercept=config["intercept"],
            regularization=config["regularization"],
            lambda_=config["lambda_"],
            l1_ratio=config["l1_ratio"],
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Step 8: Train model
    logger.info("Training model")
    model.fit(X_train_transformed, y_train)

    # Step 9: Test model
    logger.info("Evaluating model")
    model_evaluator = ModelEvaluator(model, X_test_transformed, y_test)

    # Step 10: Evaluation & Analysis of performance
    if config["model_type"] == "regression":
        perf_metrics = model_evaluator.evaluate_regression()
        perfs.append(perf_metrics)
        logger.info(f"Model performance: {perf_metrics}")

    # Step 11: Export model, pipeline, data, metrics, and logs
    logger.info("Exporting model, pipeline, data, metrics, and logs")
    exporter = Exporter(
        model=model,
        pipeline=data_processor.preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_path=OUTPUT_FOLDER_PATH,
    )
    exporter.export_model()
    exporter.export_pipeline()
    exporter.export_data()
    exporter.export_metrics(perf_metrics)

# Calculate total execution time
end_time = datetime.now()
execution_time = end_time - start_time
logger.info(f"Pipeline execution completed in {execution_time}.")
