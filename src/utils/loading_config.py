import yaml  # type: ignore


# Load configuration
def load_config(config_path: str) -> dict:
    """
    Loads the configuration file for model parameters and paths.

    Parameters:
    - config_path (str): Path to the configuration YAML file.

    Returns:
    - dict: Configuration parameters.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
