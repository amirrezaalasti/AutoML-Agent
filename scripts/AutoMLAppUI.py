import streamlit as st
import pandas as pd
import zipfile
import io
from datetime import datetime

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

from configs.api_keys import GROQ_API_KEY, GOOGLE_API_KEY
from scripts.LLMClient import LLMClient
from scripts.AutoMLAgent import AutoMLAgent

# Available GROQ models
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
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
}


class AutoMLAppUI:
    def __init__(self):
        self.dataset = None
        self.data_type = None
        self.model_choice = None
        self.dataset_name = None

    def load_dataset(self, data_type, dataset_name):
        # 1. Tabular
        if data_type == "tabular":
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

        self.data_type = st.selectbox("Select the dataset type", list(DATASET_OPTIONS.keys()))
        dataset_choice = st.selectbox("Choose a dataset", DATASET_OPTIONS[self.data_type])
        self.dataset = self.load_dataset(self.data_type, dataset_choice)

        self.model_choice = st.selectbox("Select a GROQ LLM Model", AVAILABLE_MODELS)

        if st.button("Run AutoML Agent"):
            if self.dataset is None:
                st.error("Please upload or select a dataset first.")
                return

            with st.spinner("Setting up AutoML Agent..."):
                if "gemini" in self.model_choice:
                    api_key = GOOGLE_API_KEY
                else:
                    api_key = GROQ_API_KEY

                llm_client = LLMClient(api_key=api_key, model_name=self.model_choice)
                self.dataset_name = dataset_choice
                agent = AutoMLAgent(
                    dataset=self.dataset,
                    llm_client=llm_client,
                    dataset_type=self.data_type,
                    ui_agent=st,
                    dataset_name=dataset_choice,
                )
                (
                    config_code,
                    scenario_code,
                    train_code,
                    loss,
                    prompts,
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
