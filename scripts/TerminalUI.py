import pandas as pd
from sklearn.datasets import fetch_openml

# Core dataset libraries
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    fetch_20newsgroups,
)
from keras.datasets import mnist, fashion_mnist
import statsmodels.api as sm
import seaborn as sns

from config.api_keys import GROQ_API_KEY, GOOGLE_API_KEY, LOCAL_LLAMA_API_KEY
from scripts.LLMClient import LLMClient
from scripts.AutoMLAgent import AutoMLAgent
from utils import format_dataset
from config.urls import BASE_URL

# Available GROQ models
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "llama-3.3-70b-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
]

# Datasets grouped by type
DATASET_OPTIONS = {
    "tabular": [
        "Iris",
        "Wine",
        "Breast Cancer",
        "Diabetes",
        "Digits",
        "R Duncan",
        "Titanic (seaborn)",
        "Custom Upload",
    ],
    "image": [
        "MNIST",
        "Fashion-MNIST",
    ],
    "time_series": ["Sunspot"],
    "text": ["20 Newsgroups"],
    "categorical": ["Breast Cancer", "Wine", "Adult Income"],
    "openml": ["OpenML Dataset"],
}

TASK_OPTIONS = ["classification", "regression", "clustering"]


class DummyUI:
    """A mock UI agent that prints to the console."""

    def subheader(self, text):
        print(f"\n--- {text} ---")

    def write(self, text):
        print(text)

    def code(self, code_string, language="python"):
        print("\n--- Code ---")
        print(code_string)
        print("--------------")

    def success(self, text):
        print(f"\n[SUCCESS] {text}")

    def error(self, text):
        print(f"\n[ERROR] {text}")

    def info(self, text):
        print(f"[INFO] {text}")

    def spinner(self, text=""):
        class Spinner:
            def __enter__(self):
                print(f"\nRunning: {text}...")

            def __exit__(self, exc_type, exc_val, exc_tb):
                print("Done.")

        return Spinner()


class TerminalUI:
    def __init__(self):
        self.dataset = None
        self.data_type = None
        self.model_choice = None
        self.dataset_name = None
        self.task_type = None

    def _get_choice(self, prompt: str, options: list) -> str:
        """Helper to get user choice from a list of options."""
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        while True:
            try:
                choice = int(input("Enter the number of your choice: "))
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def load_dataset(self, data_type, dataset_name):
        if data_type == "openml":
            openml_dataset_type = self._get_choice(
                "Select the type of OpenML dataset",
                ["tabular", "image", "text", "time_series", "categorical"],
            )

            fetch_method = self._get_choice(
                "Choose how to fetch the dataset:",
                ["By Name", "By ID"],
            )

            if fetch_method == "By Name":
                dataset_name = input("Enter OpenML Dataset Name (e.g., 'iris'): ")
                dataset_id = None
            else:
                dataset_id = input("Enter OpenML Dataset ID (e.g., 61): ")
                try:
                    dataset_id = int(dataset_id)
                except ValueError:
                    print("Invalid ID. Please enter a number.")
                    return None
                dataset_name = None

            if (dataset_name and dataset_name.strip()) or dataset_id:
                try:
                    # Load the dataset from OpenML
                    if dataset_id:
                        print(f"Fetching dataset ID {dataset_id} from OpenML...")
                        X, y = fetch_openml(data_id=dataset_id, return_X_y=True)
                        dataset_identifier = f"ID: {dataset_id}"
                    else:
                        print(f"Fetching '{dataset_name}' from OpenML...")
                        X, y = fetch_openml(dataset_name, return_X_y=True)
                        dataset_identifier = f"Name: {dataset_name}"

                    if y is None:
                        print("No target column found in the dataset.")
                        return None

                    # Format the dataset
                    self.dataset = format_dataset({"X": X, "y": y})
                    self.data_type = openml_dataset_type
                    self.dataset_name = dataset_name if dataset_name else f"dataset_{dataset_id}"

                    # Display dataset info
                    print(f"Dataset loaded successfully: {dataset_identifier}")
                    print(f"Number of features: {X.shape[1]}")
                    print(f"Number of instances: {X.shape[0]}")

                    return self.dataset
                except Exception as e:
                    print(f"Error loading OpenML dataset: {e}")
                    return None
            return None

        # Simplified dataset loading logic from AutoMLAppUI
        elif data_type == "tabular":
            if dataset_name == "Iris":
                X, y = load_iris(return_X_y=True, as_frame=True)
            elif dataset_name == "Wine":
                X, y = load_wine(return_X_y=True, as_frame=True)
            elif dataset_name == "Breast Cancer":
                X, y = load_breast_cancer(return_X_y=True, as_frame=True)
            elif dataset_name == "Diabetes":
                X, y = load_diabetes(return_X_y=True, as_frame=True)
            elif dataset_name == "Digits":
                X, y = load_digits(return_X_y=True, as_frame=True)
            elif dataset_name == "R Duncan":
                ds = sm.datasets.get_rdataset("Duncan", "carData").data
                X = ds.drop(columns=["prestige"])
                y = ds["prestige"]
            elif dataset_name == "Titanic (seaborn)":
                df = sns.load_dataset("titanic").dropna(subset=["survived"])
                X = df.drop(columns=["survived"])
                y = df["survived"]
            elif dataset_name == "Custom Upload":
                file_path = input("Enter the path to your CSV file: ")
                try:
                    df = pd.read_csv(file_path)
                    target_column = self._get_choice("Select the target column:", df.columns.tolist())
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                except Exception as e:
                    print(f"Error loading custom file: {e}")
                    return None
            return {"X": X, "y": y}

        elif data_type == "image":
            if dataset_name == "MNIST":
                (xtr, ytr), _ = mnist.load_data()
            elif dataset_name == "Fashion-MNIST":
                (xtr, ytr), _ = fashion_mnist.load_data()
            X = pd.DataFrame(xtr.reshape(xtr.shape[0], -1))
            y = pd.Series(ytr.flatten())
            return {"X": X, "y": y}

        elif data_type == "time_series":
            ds = sm.datasets.get_rdataset("sunspot.year", "datasets").data
            X = pd.DataFrame(ds["time"])
            y = pd.Series(ds["sunspot.year"])
            return {"X": X, "y": y}

        elif data_type == "text":
            data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
            return {"X": pd.Series(data.data), "y": pd.Series(data.target)}

        elif data_type == "categorical":
            if dataset_name == "Adult Income":
                try:
                    ds = sm.datasets.get_rdataset("adult", "OpenML").data
                    X = ds.drop(columns=["class"])
                    y = ds["class"]
                    return {"X": X, "y": y}
                except Exception as e:
                    print(f"Error loading Adult Income dataset: {e}")
                    print("Adult Income dataset not available, falling back to tabular dataset...")
                    return self.load_dataset("tabular", dataset_name)
            else:
                return self.load_dataset("tabular", dataset_name)

        return None

    def run(self):
        print("--- Welcome to the AutoML Agent Terminal UI ---")

        # 1. Select dataset type
        self.data_type = self._get_choice("Select the dataset type", list(DATASET_OPTIONS.keys()))

        # 2. Choose dataset
        self.dataset_name = self._get_choice("Choose a dataset", DATASET_OPTIONS[self.data_type])

        # 3. Load dataset
        self.dataset = self.load_dataset(self.data_type, self.dataset_name)
        if self.dataset is None:
            print("Failed to load dataset. Exiting.")
            return

        # 4. Select task type
        self.task_type = self._get_choice("Select the ML task", TASK_OPTIONS)

        # 5. Select model
        self.model_choice = self._get_choice("Select an LLM Model", AVAILABLE_MODELS)

        print("\n--- Starting AutoML Agent ---")

        base_url = None
        if "gemini" in self.model_choice:
            api_key = GOOGLE_API_KEY
        elif self.model_choice == "llama-3.3-70b-instruct":
            api_key = LOCAL_LLAMA_API_KEY
            base_url = BASE_URL
        else:
            api_key = GROQ_API_KEY

        ui_agent = DummyUI()

        agent = AutoMLAgent(
            dataset=self.dataset,
            dataset_type=self.data_type,
            ui_agent=ui_agent,
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            model_name=self.model_choice,
            api_key=api_key,
            base_url=base_url,
        )

        (
            config_code,
            scenario_code,
            train_function_code,
            last_loss,
            metrics,
            prompts,
            experiment_dir,
        ) = agent.generate_components()

        print("\n--- AutoML Agent finished ---")
        print(f"Final loss: {last_loss}")
        print(f"Results saved to: {experiment_dir}")
