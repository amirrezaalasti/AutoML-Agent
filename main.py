from sklearn.datasets import load_iris
from scripts.AutoMLAgent import AutoMLAgent
from ConfigSpace import ConfigurationSpace
from configs.api_keys import GROQ_API_KEY
from scripts.LLMClient import LLMClient
import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_openml
from keras.datasets import mnist

# Available GROQ models
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "whisper-large-v3",
    "distil-whisper-large-v3-en",
]

# Available Dataset Types
DATASET_TYPES = ["tabular", "time_series", "image", "text", "categorical"]


def generate_response():
    llm = LLMClient(api_key=GROQ_API_KEY)
    response = llm.generate(
        prompt="Generate a configuration space for a random forest classifier with max_depth between 1 and 10.",
        model="llama-3.3-70b-versatile",
    )
    print(response)


X, y = load_iris(return_X_y=True)
# agent.generate_components()
# incumbent = agent.run_smac()
# print(f"Best configuration: {incumbent}")

if __name__ == "__main__":
    st.title("AutoML Agent Interface")

    dataset_option = st.selectbox(
        "Choose a dataset", ("Use Iris Dataset", "MNIST Dataset", "Upload CSV File")
    )

    if dataset_option == "Use Iris Dataset":
        X, y = load_iris(return_X_y=True)
        dataset = {"X": X, "y": y}
    elif dataset_option == "MNIST Dataset":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        # Combine train and test sets
        X = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], ignore_index=True)
        y = pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)
        dataset = {"X": X, "y": y}
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            target_column = st.selectbox(
                "Select the target column (label):", df.columns
            )
            X = df.drop(columns=[target_column])
            y = df[target_column]
            dataset = {"X": X, "y": y}
        else:
            dataset = None
    # 2. Dataset Type
    data_type = st.selectbox("Select the dataset type", DATASET_TYPES)

    # 3. LLM Model
    model_choice = st.selectbox("Select a GROQ LLM Model", AVAILABLE_MODELS)

    # 4. Run Agent
    if st.button("Run AutoML Agent"):
        if dataset is None:
            st.error("Please upload a dataset first.")
        else:
            with st.spinner("Setting up AutoML Agent..."):
                llm_client = LLMClient(api_key=GROQ_API_KEY, model_name=model_choice)
                agent = AutoMLAgent(
                    dataset=dataset, llm_client=llm_client, dataset_type=data_type
                )
                config_code, scenario_code, train_code, loss, prompts = (
                    agent.generate_components()
                )
                st.success("AutoML Agent setup complete!")
                st.subheader("Generated Configuration Space Code")
                st.code(config_code, language="python")
                st.subheader("Generated Scenario Code")
                st.code(scenario_code, language="python")
                st.subheader("Generated Training Function Code")
                st.code(train_code, language="python")
                st.subheader("Loss Value")
                st.write(loss)
                st.subheader("Prompts Used")
                st.write(prompts)
                # # incumbent = agent.run_smac()
                # # print(f"Best configuration: {incumbent}")
