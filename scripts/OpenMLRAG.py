import openml
from configs.api_keys import OPENML_API_KEY
import pandas as pd


class OpenMLRAG:
    def __init__(self):
        openml.config.apikey = OPENML_API_KEY

    def search_datasets(self, keyword: str) -> pd.DataFrame:
        """
        Searches for active OpenML datasets by a flexible keyword search in their name.

        This function splits the keyword into individual words and finds datasets
        whose names contain ALL of those words, regardless of what's between them.

        Args:
            keyword: The string to search for (e.g., "breast cancer", "kr kp").

        Returns:
            A pandas DataFrame containing all matching datasets, sorted by relevance.
        """
        # Fetch all active datasets from OpenML
        datasets_df = openml.datasets.list_datasets(output_format="dataframe", status="active")

        # Split the search query into individual words
        search_words = str(keyword).lower().split()

        if not search_words:
            # If the keyword is empty or just spaces, return an empty DataFrame
            return pd.DataFrame()

        # Start with a mask that includes all datasets
        final_mask = pd.Series(True, index=datasets_df.index)

        # Sequentially filter the DataFrame for each word in the search query
        # This ensures the name contains 'word1' AND 'word2' AND ...
        for word in search_words:
            final_mask = final_mask & datasets_df["name"].str.lower().str.contains(word)

        related_datasets = datasets_df[final_mask]

        # Sort results by a relevance metric.
        return related_datasets.sort_values(by="NumberOfInstances", ascending=False)

    def get_related_datasets(self, tag=None, dataset_name=None):
        """
        Get datasets related to a specific tag or dataset name.
        If dataset_name is provided, it will return datasets related to that specific dataset.
        If tag is provided, it will return datasets related to that tag.
        """
        datasets = openml.datasets.list_datasets(output_format="dataframe", status="active")
        if dataset_name:
            # choose the first dataset that matches the name
            datasets = self.search_datasets(dataset_name)
            return datasets.iloc[0] if not datasets.empty else None
        elif tag:
            return openml.datasets.list_datasets(tag=tag)
        else:
            return datasets

    def get_related_tasks_of_dataset(self, dataset_id, task_type=None):
        """
        Get tasks related to a specific dataset ID.
        If task_type is provided, it will filter tasks by that type.
        """
        tasks = openml.tasks.list_tasks(data_id=dataset_id, output_format="dataframe")
        # if the lower_case ttid includes the task_type, then return it
        if task_type:
            tasks = tasks[tasks["ttid"].astype(str).str.lower().str.contains(task_type.lower())]

        return tasks[:3] if len(tasks) > 10 else tasks

    def get_setup_parameters_of_tasks(self, task_ids, function="predictive_accuracy"):
        """
        Get the top runs of a specific task based on a given function.
        """
        evaluations = openml.evaluations.list_evaluations(
            tasks=task_ids,
            function=function,
            output_format="dataframe",
            sort_order="desc",
        )
        top_evaluations = evaluations.head(1)

        setup_parameters = []
        for evaluation in top_evaluations["setup_id"].unique():
            setup = openml.setups.get_setup(evaluation)
            setup_parameters.append(setup.parameters)

        return setup_parameters
