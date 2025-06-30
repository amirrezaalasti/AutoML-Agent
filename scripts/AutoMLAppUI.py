import streamlit as st
import pandas as pd
import zipfile
import io
from datetime import datetime
import os
from sklearn.datasets import fetch_openml
import openml

# Core dataset libraries (conflict‑free)
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    fetch_20newsgroups,
)
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import statsmodels.api as sm
import seaborn as sns

from config.api_keys import GROQ_API_KEY, GOOGLE_API_KEY, LOCAL_LLAMA_API_KEY
from scripts.LLMClient import LLMClient
from scripts.AutoMLAgent import AutoMLAgent
from scripts.utils import convert_to_csv, format_dataset
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

# Datasets grouped by type (conflict‑free)
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
        "Fashion‑MNIST",
    ],
    "time_series": ["Sunspot"],
    "text": ["20 Newsgroups"],
    "categorical": ["Breast Cancer", "Wine", "Adult Income"],
    "openml": ["OpenML Dataset"],
}

TASK_OPTIONS = ["classification", "regression", "clustering"]


class AutoMLAppUI:
    def __init__(self):
        self.dataset = None
        self.data_type = None
        self.model_choice = None
        self.dataset_name = None
        self.task_type = None

    def load_dataset(self, data_type, dataset_name):
        if data_type == "openml":
            # First, let user choose the dataset type
            openml_dataset_type = st.selectbox(
                "Select the type of OpenML dataset",
                ["tabular", "image", "text", "time_series", "categorical"],
                key="openml_dataset_type",
            )

            # Add option to choose between name and ID
            fetch_method = st.radio(
                "Choose how to fetch the dataset:",
                ["By Name", "By ID"],
                key="fetch_method",
            )

            if fetch_method == "By Name":
                # Then, let user enter the dataset name
                dataset_name = st.text_input("Enter OpenML Dataset Name (e.g., iris)", key="openml_dataset_name")
                dataset_id = None
            else:
                # Let user enter the dataset ID
                dataset_id = st.number_input(
                    "Enter OpenML Dataset ID (e.g., 61)",
                    min_value=1,
                    key="openml_dataset_id",
                )
                dataset_name = None

            if (dataset_name and dataset_name.strip()) or dataset_id:
                try:
                    # Load the dataset from OpenML
                    if dataset_id:
                        X, y = fetch_openml(data_id=dataset_id, return_X_y=True)
                        dataset_identifier = f"ID: {dataset_id}"
                    else:
                        X, y = fetch_openml(dataset_name, return_X_y=True)
                        dataset_identifier = f"Name: {dataset_name}"

                    if y is None:
                        st.error("No target column found in the dataset.")

                    # Format the dataset
                    self.dataset = format_dataset({"X": X, "y": y})
                    self.data_type = openml_dataset_type
                    self.dataset_name = openml.datasets.get_dataset(dataset_id).name

                    # Display dataset info
                    st.info(f"Dataset loaded successfully: {dataset_identifier}")
                    st.info(f"Number of features: {X.shape[1]}")
                    st.info(f"Number of instances: {X.shape[0]}")

                    return self.dataset
                except Exception as e:
                    st.error(f"Error loading OpenML dataset: {str(e)}")
                    return None
            return None
        # 1. Tabular
        elif data_type == "tabular":
            if dataset_name == "Iris":
                X, y = load_iris(return_X_y=True)
                return {"X": X, "y": y}

            elif dataset_name == "Wine":
                X, y = load_wine(return_X_y=True)
                return {"X": X, "y": y}

            elif dataset_name == "Breast Cancer":
                X, y = load_breast_cancer(return_X_y=True)
                return {"X": X, "y": y}

            elif dataset_name == "Diabetes":
                X, y = load_diabetes(return_X_y=True)
                return {"X": X, "y": y}

            elif dataset_name == "Digits":
                X, y = load_digits(return_X_y=True)
                return {"X": X, "y": y}

            elif dataset_name == "R Duncan":
                ds = sm.datasets.get_rdataset("Duncan", "carData").data
                X = ds.drop(columns=["prestige"])
                y = ds["prestige"]
                return {"X": X, "y": y}

            elif dataset_name == "Titanic (seaborn)":
                df = sns.load_dataset("titanic").dropna(subset=["survived"])
                X = df.drop(columns=["survived"])
                y = df["survived"]
                return {"X": X, "y": y}

            elif dataset_name == "Custom Upload":
                uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    target_column = st.selectbox("Select the target column (label):", df.columns)
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                    return {"X": X, "y": y}
                return None

        # 2. Image (only Keras)
        elif data_type == "image":
            if dataset_name == "MNIST":
                (xtr, ytr), _ = mnist.load_data()
            elif dataset_name == "Fashion‑MNIST":
                (xtr, ytr), _ = fashion_mnist.load_data()
            elif dataset_name == "CIFAR‑10":
                (xtr, ytr), _ = cifar10.load_data()
            elif dataset_name == "CIFAR‑100":
                (xtr, ytr), _ = cifar100.load_data()
            # flatten images
            X = pd.DataFrame(xtr.reshape(xtr.shape[0], -1))
            y = pd.Series(ytr)
            return {"X": X, "y": y}

        # 3. Time Series
        elif data_type == "time_series":
            ds = sm.datasets.get_rdataset("sunspot.year", "datasets").data
            return {"X": ds.index.values.reshape(-1, 1), "y": ds.value}

        # 4. Text
        elif data_type == "text":
            data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
            return {"X": pd.Series(data.data), "y": pd.Series(data.target)}

        # 5. Categorical
        elif data_type == "categorical":
            if dataset_name == "Adult Income":
                df = sm.datasets.get_rdataset("adult", "OpenML").data
                X = df.drop(columns=["class"])
                y = df["class"]
                return {"X": X, "y": y}
            return self.load_dataset("tabular", dataset_name)

        return None

    def display(self):
        st.title("AutoML Agent Interface")

        # Step 1: Select dataset type
        self.data_type = st.selectbox(
            "Select the dataset type",
            list(DATASET_OPTIONS.keys()),
            key="main_dataset_type",
        )

        # Step 2: Choose dataset
        dataset_choice = st.selectbox("Choose a dataset", DATASET_OPTIONS[self.data_type], key="dataset_choice")

        # Step 3: Load dataset
        self.dataset = self.load_dataset(self.data_type, dataset_choice)

        # Step 4: Select task type (for all datasets)
        self.task_type = st.selectbox("Select the ML task for this dataset", TASK_OPTIONS, key="task_type")

        # Step 5: Select model
        self.model_choice = st.selectbox("Select a GROQ LLM Model", AVAILABLE_MODELS, key="model_choice")

        if st.button("Run AutoML Agent"):
            if self.dataset is None:
                st.error("Please upload or select a dataset first.")
                return

            with st.spinner("Setting up AutoML Agent..."):
                base_url = None
                if "gemini" in self.model_choice:
                    api_key = GOOGLE_API_KEY
                elif self.model_choice == "llama-3.3-70b-instruct":
                    api_key = LOCAL_LLAMA_API_KEY
                    base_url = BASE_URL
                else:
                    api_key = GROQ_API_KEY

                if dataset_choice != "OpenML Dataset":
                    self.dataset_name = dataset_choice

                agent = AutoMLAgent(
                    dataset=self.dataset,
                    dataset_type=self.data_type,
                    ui_agent=st,
                    dataset_name=self.dataset_name,
                    task_type=self.task_type,
                    model_name=self.model_choice,
                    api_key=api_key,
                    base_url=base_url,
                )
                (
                    config_code,
                    scenario_code,
                    train_code,
                    loss,
                    prompts,
                    logger_dir,
                ) = agent.generate_components()

                st.subheader("Prompts Used")
                st.write(prompts)

                # Add download button for generated code and prompts
                zip_buffer = self.create_download_zip(config_code, scenario_code, train_code, prompts)

                st.download_button(
                    label="Download Generated Code and Prompts",
                    data=zip_buffer,
                    file_name=f"automl_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                )

    def create_download_zip(self, config_code, scenario_code, train_code, prompts):
        """Create a zip file containing all generated code and prompts."""
        zip_buffer = io.BytesIO()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add code files
            zip_file.writestr("config.py", config_code)
            zip_file.writestr("scenario.py", scenario_code)
            zip_file.writestr("train.py", train_code)

            # Add the requirements.txt file
            with open("requirements.txt", "r") as req_file:
                zip_file.writestr("requirements.txt", req_file.read())

            # add the dataset
            # Convert DataFrame to CSV string before writing
            convert_to_csv(self.dataset)
            zip_file.write("dataset.csv", "dataset.csv")
            zip_file.write("target.csv", "target.csv")

            # Add prompts
            zip_file.writestr("prompts.txt", "\n\n".join(prompts))

            # Add a README
            readme_content = f"""AutoML Generated Code
            Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Dataset: {self.data_type} - {self.dataset_name}
            Model: {self.model_choice}

            This zip contains:
            - config.py: Configuration code
            - scenario.py: Scenario implementation
            - train.py: Training code
            - prompts.txt: All prompts used during generation
            """
            zip_file.writestr("README.txt", readme_content)

        zip_buffer.seek(0)
        return zip_buffer
