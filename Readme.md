# Agent SMAC

<img src="logo.png">

**AutoML-Agent** is a prototype framework designed to automate the end-to-end machine learning pipeline using a multi-agent system powered by Large Language Models (LLMs). This project aims to simplify the process of building, training, and deploying machine learning models by interpreting natural language instructions and orchestrating specialized agents for each stage of the pipeline.

## 🚀 Features

* **Natural Language Interface**: Accepts user-defined task descriptions to initiate the AutoML process.
* **Multi-Agent Collaboration**: Decomposes tasks into subtasks handled by specialized LLM agents (e.g., data preprocessing, model selection, training).
* **Parallel Execution**: Executes subtasks concurrently to enhance efficiency.
* **Modular Design**: Facilitates easy integration of new agents or modification of existing ones.
* **Extensible Framework**: Designed to support various data modalities and machine learning tasks.

## 🧠 Architecture Overview

The AutoML-Agent framework operates through the following stages:

1. **Initialization**: Parses and validates user instructions.
2. **Planning**: Breaks down the overall task into manageable subtasks.
3. **Execution**: Assigns subtasks to specialized agents for processing.
4. **Verification**: Evaluates the outputs of each agent to ensure correctness.
5. **Deployment**: Aggregates the results into a deployable machine learning model.

## 📁 Repository Structure

```
AutoML-Agent/
├── automl_results/             # Outputs and results from AutoML runs
├── logs/                       # Log files for monitoring and debugging
├── notebooks/                  # Jupyter notebooks for experimentation
├── scripts/                    # Utility scripts for various tasks
├── templates/                  # Templates for reports or model deployment
├── .gitignore                  # Specifies files to ignore in version control
├── Readme.md                   # Project documentation
├── Streamlit.pdf               # Streamlit application guide or documentation
├── functionality-gif.mov       # Demonstration of project functionality
├── main.py                     # Main script to run the AutoML-Agent
└── test.ipynb                  # Test notebook for validating functionalities
```

## 🛠️ Getting Started

### Prerequisites

* Python 3.8 or higher
* [pip](https://pip.pypa.io/en/stable/) package manager

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/amirrezaalasti/AutoML-Agent.git
   cd AutoML-Agent
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure that the `requirements.txt` file is present in the root directory with all necessary dependencies listed.*

### Running the Application

To initiate the AutoML process:

```bash
python main.py
```

Follow the on-screen prompts to input your task description and configure settings as needed.

## 🧪 Example Usage

*Example:*

Suppose you have a CSV file containing customer data and you want to predict customer churn.

1. **Prepare your dataset** and place it in the appropriate directory.

2. **Run the application:**

   ```bash
   python main.py
   ```

3. **Input your task description** when prompted:

   ```
   Predict customer churn based on the provided dataset.
   ```

The AutoML-Agent will process your request, perform data preprocessing, select suitable models, train and evaluate them, and finally provide a deployable model along with performance metrics.

## 📊 Supported Tasks

* **Classification**: Binary and multi-class classification tasks.
* **Regression**: Predicting continuous values.
* **Clustering**: Grouping similar data points.
* **Natural Language Processing**: Text classification, sentiment analysis, etc.
* **Computer Vision**: Image classification, object detection, etc.

*Note: The current prototype primarily supports tabular data. Support for other data modalities is under development.*

## 📈 Performance and Benchmarks

*Details about model performance, benchmarks on standard datasets, and comparisons with other AutoML tools can be added here.*

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository.

2. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and commit them:

   ```bash
   git commit -m "Add your message here"
   ```

4. Push to your forked repository:

   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a pull request detailing your changes.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software in accordance with the license terms.

## 📬 Contact

For questions, suggestions, or collaborations, please contact:

* **Amirreza Alasti**
  [GitHub Profile](https://github.com/amirrezaalasti)

---

*Disclaimer: This is a prototype project and is currently under active development. Features and functionalities are subject to change.*
