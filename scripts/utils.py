def describe_dataset(dataset):
    """
    Describe the dataset in a human-readable format.
    
    Args:
        dataset (dict): A dictionary containing 'X' and 'y' keys.
    
    Returns:
        str: A description of the dataset.
    """
    X, y = dataset['X'], dataset['y']
    num_samples, num_features = X.shape
    num_classes = len(set(y))
    
    description = (
        f"The dataset contains {num_samples} samples and {num_features} features.\n"
        f"There are {num_classes} unique classes in the target variable."
        f"\nThe features are of type {X.dtype} and the target variable is of type {y.dtype}.\n"
        f"The first few samples of the features are:\n{X[:5]}\n"
        f"The first few samples of the target variable are:\n{y[:5]}\n"
        f"The feature names are: {X.columns.tolist() if hasattr(X, 'columns') else 'N/A'}\n"
    )
    
    return description

def log_message(message: str, log_file: str = "./logs/logs.txt"):
    """
    Log a message to a specified log file.
    
    Args:
        message (str): The message to log.
        log_file (str): The path to the log file. Default is "./logs/logs.txt".
    """
    with open(log_file, "a") as log:
        log.write(message + "\n")

def save_code(code: str, filename: str):
    """
    Save the generated code to a file.
    
    Args:
        code (str): The code to save.
        filename (str): The name of the file to save the code in.
    """
    with open(filename, "w") as file:
        file.write(code)