import openml
import pandas as pd
import numpy as np
from amltk.metalearning import dataset_distance
from collections import OrderedDict

# This should be configured securely, e.g., via environment variables
# from configs.api_keys import OPENML_API_KEY


class OpenMLRAG:
    """
    A class to interact with OpenML for finding similar datasets and suggesting
    hyperparameter configurations based on meta-learning principles.
    """

    def __init__(self, openml_api_key: str, metafeatures_csv_path: str):
        """
        Initializes the OpenMLRAG helper.

        Args:
            openml_api_key: Your API key for OpenML.
            metafeatures_csv_path: The file path to your pre-generated CSV
                                   of OpenML meta-features.
        """
        openml.config.apikey = openml_api_key
        self.metafeatures_df = pd.read_csv(metafeatures_csv_path, index_col="did")

    def search_datasets(self, keyword: str) -> pd.DataFrame:
        """
        Searches for active OpenML datasets by a keyword in their name using the local CSV.
        """
        if not keyword or keyword.isspace():
            return pd.DataFrame()

        # Use the pre-loaded DataFrame for a fast, local search
        search_results = self.metafeatures_df[self.metafeatures_df["name"].str.contains(keyword, case=False, na=False)]
        return search_results

    def get_source_dataset(self, dataset_name: str) -> pd.Series | None:
        """
        Finds a single source dataset from the local CSV based on a name.
        """
        datasets = self.search_datasets(dataset_name)
        # Return the first search result if any are found
        return datasets.iloc[0] if not datasets.empty else None

    def find_similar_datasets(self, source_dataset_id: int, n_similar: int = 5) -> pd.DataFrame:
        """
        Finds datasets similar to a source dataset using the FAST local CSV method.
        """
        if source_dataset_id not in self.metafeatures_df.index:
            raise ValueError(f"Source dataset ID {source_dataset_id} not found in the local CSV.")

        source_metafeatures = self.metafeatures_df.loc[source_dataset_id]
        other_metafeatures_df = self.metafeatures_df.drop(source_dataset_id)

        # Align columns and select only numeric features for distance calculation
        common_features = source_metafeatures.index.intersection(other_metafeatures_df.columns)
        source_aligned = source_metafeatures[common_features]
        others_aligned = other_metafeatures_df[common_features]

        numeric_cols = others_aligned.select_dtypes(include=np.number).columns
        source_numeric = source_aligned[numeric_cols].dropna()
        others_numeric = others_aligned[numeric_cols].dropna(axis=1)

        final_common_features = source_numeric.index.intersection(others_numeric.columns)
        source_final = source_numeric[final_common_features]
        others_final = others_numeric[final_common_features]

        # Calculate distances to find the most similar datasets
        distances = dataset_distance(
            target=source_final,
            dataset_metafeatures=others_final.T.to_dict("series"),
            distance_metric="l2",
            scaler="minmax",
            closest_n=n_similar,
        )

        # Return the metadata for the most similar datasets from our local CSV
        return self.metafeatures_df.loc[distances.index]

    def get_related_tasks_of_dataset(
        self,
        dataset_id: int,
        task_types: list[str] = ["classification"],
        n_tasks: int = 3,
    ) -> pd.DataFrame:
        """
        Gets a sample of top tasks related to a specific dataset ID.
        """
        # --- MINOR IMPROVEMENT: Use task_type parameter directly for cleaner code ---
        tasks = openml.tasks.list_tasks(data_id=dataset_id, output_format="dataframe")
        if task_types:
            # get tasks that have one of the task types
            for task_type in task_types:
                tasks = tasks[tasks["ttid"].astype(str).str.lower().str.contains(task_type.lower())]
                if not tasks.empty:
                    # task_types is a list of task types, so we need to get the first task type that is found
                    break
        return tasks.head(n_tasks)

    def get_top_setups_for_tasks(
        self,
        task_ids: list[int],
        function: str = "predictive_accuracy",
        n_setups: int = 3,
    ) -> list[dict]:
        """
        Gets the top performing setups (algorithm + hyperparameters) for a list of tasks.
        """
        if not task_ids:
            return []

        evaluations = openml.evaluations.list_evaluations(
            tasks=task_ids,
            function=function,
            output_format="dataframe",
            sort_order="desc",
        )

        # --- FIX: The main reason for the error. Handle cases where no evaluations are found. ---
        if evaluations.empty:
            print(f"   -> No evaluations found for tasks: {task_ids}")
            return []
        # --- End of FIX ---

        top_evaluations = evaluations.head(n_setups)

        setups = []
        # Use unique setup IDs to avoid fetching the same setup multiple times
        for setup_id in top_evaluations["setup_id"].unique():
            setup = openml.setups.get_setup(setup_id)
            # Find the corresponding flow name and performance from the evaluation table
            evaluation_row = top_evaluations[top_evaluations["setup_id"] == setup_id].iloc[0]
            params = {p.parameter_name: p.value for p in setup.parameters.values()}

            setups.append(
                {
                    "flow_name": evaluation_row["flow_name"],
                    "performance": round(evaluation_row["value"], 4),
                    "hyperparameters": params,
                }
            )
        return setups

    def extract_suggested_config_space_parameters(self, dataset_name_in_openml: str, task_types: list[str] = ["classification"]) -> list[dict]:
        """
        Extracts a clean list of suggested hyperparameter configurations from datasets
        similar to the provided one.
        :param dataset_name_in_openml: The name of the dataset in OpenML to find related datasets.
        :param task_types: The types of tasks to search for. the order of the list is important. the first task type is the most important.
        :return: A list of suggested hyperparameter configurations.
        """
        print(f"1. Searching for source dataset: '{dataset_name_in_openml}'...")
        source_dataset = self.get_source_dataset(dataset_name=dataset_name_in_openml)
        if source_dataset is None:
            print(f"❌ Error: No datasets found for '{dataset_name_in_openml}'.")
            return []

        source_id = int(source_dataset.name)
        print(f"   ✅ Found '{source_dataset['name']}' with ID: {source_id}")

        print(f"\n2. Finding the top 3 datasets similar to '{source_dataset['name']}'...")
        similar_datasets = self.find_similar_datasets(source_dataset_id=source_id, n_similar=3)
        print("   ✅ Found similar datasets:", similar_datasets["name"].tolist())

        print("\n3. Gathering top-performing setups from these similar datasets...")
        all_setups = []
        for dataset_id, dataset_row in similar_datasets.iterrows():
            # --- MINOR IMPROVEMENT: Added a print statement for better logging ---
            print(f" -> Processing similar dataset: '{dataset_row['name']}' (ID: {dataset_id})")
            tasks_df = self.get_related_tasks_of_dataset(dataset_id, task_types=task_types)
            task_ids = tasks_df["tid"].tolist()
            if task_ids:
                setups = self.get_top_setups_for_tasks(task_ids)
                all_setups.extend(setups)

        print(f"\n   ✅ Gathered {len(all_setups)} total setups.")

        print("\n4. Cleaning and simplifying the final list for the LLM...")
        # This step extracts only the hyperparameter dictionaries
        final_hyperparameters = [setup["hyperparameters"] for setup in all_setups]

        # De-duplicate the list of dictionaries
        # Dictionaries are not hashable, so we convert them to a hashable form (tuple of items)
        unique_configs = list(OrderedDict((tuple(sorted(d.items())), d) for d in final_hyperparameters).values())
        print(f"   ✅ Final unique configurations found: {len(unique_configs)}")
        return unique_configs
