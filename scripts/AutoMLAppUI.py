import streamlit as st
import pandas as pd

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
        "R Duncan (statsmodels)",
        "Titanic (seaborn)",
        "Custom Upload",
    ],
    "image": [
        "MNIST (Keras)",
        "Fashion‑MNIST (Keras)",
    ],
    "time_series": ["Sunspots (statsmodels)"],
    "text": ["20 Newsgroups"],
    "categorical": ["Breast Cancer", "Wine", "Adult Income (statsmodels)"],
}


class AutoMLAppUI:
    def __init__(self):
        self.dataset = None
        self.data_type = None
        self.model_choice = None

    def load_dataset(self, data_type, dataset_name):
        # 1. Tabular
        if data_type == "tabular":
            if dataset_name == "Iris":
                X, y = load_iris(return_X_y=True)
                return {"X": X, "y": y}

            if dataset_name == "Wine":
                X, y = load_wine(return_X_y=True)
                return {"X": X, "y": y}

            if dataset_name == "Breast Cancer":
                X, y = load_breast_cancer(return_X_y=True)
                return {"X": X, "y": y}

            if dataset_name == "Diabetes":
                X, y = load_diabetes(return_X_y=True)
                return {"X": X, "y": y}

            if dataset_name == "Digits":
                X, y = load_digits(return_X_y=True)
                return {"X": X, "y": y}

            if dataset_name == "R Duncan (statsmodels)":
                ds = sm.datasets.get_rdataset("Duncan", "carData").data
                X = ds.drop(columns=["prestige"])
                y = ds["prestige"]
                return {"X": X, "y": y}

            if dataset_name == "Titanic (seaborn)":
                df = sns.load_dataset("titanic").dropna(subset=["survived"])
                X = df.drop(columns=["survived"])
                y = df["survived"]
                return {"X": X, "y": y}

            if dataset_name == "Custom Upload":
                uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    target_column = st.selectbox("Select the target column (label):", df.columns)
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                    return {"X": X, "y": y}
                return None

        # 2. Image (only Keras)
        if data_type == "image":
            if dataset_name == "MNIST (Keras)":
                (xtr, ytr), _ = mnist.load_data()
            elif dataset_name == "Fashion‑MNIST (Keras)":
                (xtr, ytr), _ = fashion_mnist.load_data()
            elif dataset_name == "CIFAR‑10 (Keras)":
                (xtr, ytr), _ = cifar10.load_data()
            elif dataset_name == "CIFAR‑100 (Keras)":
                (xtr, ytr), _ = cifar100.load_data()
            # flatten images
            X = pd.DataFrame(xtr.reshape(xtr.shape[0], -1))
            y = pd.Series(ytr)
            return {"X": X, "y": y}

        # 3. Time Series
        if data_type == "time_series":
            ds = sm.datasets.get_rdataset("sunspot.year", "datasets").data
            return {"X": ds.index.values.reshape(-1, 1), "y": ds.value}

        # 4. Text
        if data_type == "text":
            data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
            return {"X": pd.Series(data.data), "y": pd.Series(data.target)}

        # 5. Categorical
        if data_type == "categorical":
            if dataset_name == "Adult Income (statsmodels)":
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
                agent = AutoMLAgent(
                    dataset=self.dataset,
                    llm_client=llm_client,
                    dataset_type=self.data_type,
                    ui_agent=st,
                )
                (
                    config_code,
                    scenario_code,
                    train_code,
                    loss,
                    prompts,
                ) = agent.generate_components()

                st.success("AutoML Agent setup complete!")
                st.subheader("Loss Value")
                st.write(loss)
                st.subheader("Prompts Used")
                st.write(prompts)
