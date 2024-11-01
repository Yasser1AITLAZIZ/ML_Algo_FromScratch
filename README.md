# ML_Algo_FromScratch

This repository contains machine learning algorithms implemented from scratch in Python. The goal is to provide a deeper understanding of the internal workings of common ML algorithms without relying on external libraries for algorithmic computation.

## About

This repository is dedicated to coding core ML algorithms from scratch to foster a fundamental understanding of their mechanics. Each algorithm is implemented in a modular format with clean code and comprehensive comments to facilitate learning.

The repository also includes a data processing pipeline, model evaluation, logging, and export functions to enable a complete machine learning workflow from data preprocessing to model evaluation and logging.

## Algorithms Implemented

The algorithms currently available in this repository include:

- **Linear Regression**: Implemented with Lasso, Ridge, and Elastic Net regularization options.

## Installation

To run the code in this repository, ensure you have Python installed, then install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
ML_Algo_FromScratch/
├── src/
│   ├── configuration/
│   │   ├── config_main.yaml                    # Main configuration file for pipeline setup
│   │   ├── environment_variables.py            # Global variables to use
│   │   ├── model_algo/                         # Contains implementations of ML algorithms
│   │   ├── data_processing/                    # Contains data processing pipeline
│   │   ├── model_evaluator/                    # Module for model evaluation metrics
│   │   ├── experiments/                        # Directory to store experiment outputs (models, logs, metrics)
│   │   └── exporter/                           # Contains code for exporting models, logs, and metrics
│   ├── utils/
│   │   ├── loading_config.py                   # Utility to load configuration files
│   │   └── logger.py                           # Logger configuration for console and file logging
│   └── main.py                                 # Main script to run the full ML pipeline
├── requirements.txt                            # Python package dependencies
└── README.md                                   # Project documentation
```

## Configuration

All configurations for data processing, model parameters, and export settings are centralized in the `src/configuration/config_main.yaml` file. You can specify parameters like:

- **Data processing**: Scaling methods (standard, min-max), one-hot encoding, etc.
- **Model parameters**: Algorithm choice (Linear Regression, Lasso, Ridge, Elastic Net), hyperparameters (learning rate, iterations, regularization strength), etc.
- **Export settings**: Paths for saving models, logs, data, and performance metrics.

### Example `config_main.yaml`

```yaml
data_path: "path/to/your/data.csv"
target_column: "target"
test_size: 0.2
random_state: 42

# Data processing settings
scaling: "minmax" # Options: "standard" or "minmax"
one_hot_encode: True # Apply one-hot encoding for categorical features
numeric_features: ["feature1", "feature2"]
categorical_features: ["feature3"]

# Model settings
models: ["linear_regression"]
alpha: 0.001
iterations: 500
regularization: "ridge" # Options: "lasso", "ridge", "elastic_net"
lambda_: 0.1
l1_ratio: 0.5 # Only for elastic net

# Export settings
output_path: "src/experiments"
```

## Usage

1. **Running the Pipeline**: To execute the entire pipeline with custom configurations specified in `config_main.yaml`, run the main script:

   ```bash
   python src/main.py
   ```

2. **Logging**: Each run generates a unique log file in the `OUTPUT_FOLDER` variable directory specified in `environment_variables.py`. Logs include informational messages on pipeline stages and any warnings or errors encountered.

3. **Output and Export**: After running the pipeline, results such as trained models, data processing pipelines, test data, and performance metrics are stored in a unique experiment directory (`experiments/experience_<timestamp>`). Each experiment directory contains:
   - `model.pkl`: The serialized model.
   - `pipeline.pkl`: The data processing pipeline used.
   - `train_data.parquet` & `test_data.parquet`: The processed training and test sets.
   - `model_metrics.csv`: Performance metrics for the model.
   - `experiment.log`: Log file for the specific run.

## Exécution des expériences

To perform an experiment with different models or parameters:

1. Modify the parameters in `config_main.yaml` as needed.
2. Run `main.py` again to create a new experiment directory with updated outputs.

## License

## Contributing
