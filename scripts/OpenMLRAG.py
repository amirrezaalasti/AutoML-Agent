import openml
import pandas as pd
import numpy as np
from amltk.metalearning import dataset_distance
from collections import OrderedDict


class OpenMLRAG:
    """
    A class to interact with OpenML for finding similar datasets and suggesting
    hyperparameter configurations based on meta-learning principles.
    """

    def __init__(self, openml_api_key: str, metafeatures_csv_path: str):
        openml.config.apikey = openml_api_key
        self.metafeatures_df = pd.read_csv(metafeatures_csv_path, index_col="did")
        # CHANGE: Proactively filter for only active datasets on initialization
        self.active_metafeatures_df = self.metafeatures_df[self.metafeatures_df["status"] == "active"].copy()

    def search_datasets(self, keyword: str) -> pd.DataFrame:
        if not keyword or keyword.isspace():
            return pd.DataFrame()
        # CHANGE: Search within the active datasets dataframe
        search_results = self.active_metafeatures_df[self.active_metafeatures_df["name"].str.contains(keyword, case=False, na=False)]
        return search_results

    def get_source_dataset(self, dataset_name: str) -> pd.Series | None:
        datasets = self.search_datasets(dataset_name)
        return datasets.iloc[0] if not datasets.empty else None

    def find_similar_datasets(self, source_dataset_id: int, n_similar: int = 3) -> pd.DataFrame:
        """
        Finds datasets similar to a source dataset, filtering for 'active' status
        and excluding all other datasets with the same name.
        """
        if source_dataset_id not in self.active_metafeatures_df.index:
            raise ValueError(f"Source dataset ID {source_dataset_id} not found in the active local CSV.")

        source_metafeatures = self.active_metafeatures_df.loc[source_dataset_id]
        source_name = source_metafeatures["name"]

        other_metafeatures_df = self.active_metafeatures_df[self.active_metafeatures_df["name"] != source_name]

        if other_metafeatures_df.empty:
            raise ValueError(f"No other active datasets found to compare with '{source_name}'.")

        common_features = source_metafeatures.index.intersection(other_metafeatures_df.columns)
        source_aligned = source_metafeatures[common_features]
        others_aligned = other_metafeatures_df[common_features]

        numeric_cols = others_aligned.select_dtypes(include=np.number).columns
        source_numeric = source_aligned[numeric_cols].dropna()
        others_numeric = others_aligned[numeric_cols].dropna(axis=1)

        final_common_features = source_numeric.index.intersection(others_numeric.columns)
        source_final = source_numeric[final_common_features]
        others_final = others_numeric[final_common_features]

        distances = dataset_distance(
            target=source_final,
            dataset_metafeatures=others_final.T.to_dict("series"),
            distance_metric="l2",
            scaler="minmax",
            closest_n=n_similar,
        )
        return self.active_metafeatures_df.loc[distances.index]

    # In your OpenMLRAG class:

    def get_top_setups_for_dataset(
        self,
        dataset_id: int,
        function: str = "predictive_accuracy",
        n_setups: int = 3,
    ) -> list[dict]:
        """
        Gets top setups for a dataset by first finding all its tasks,
        then getting the best evaluations for those tasks.
        """
        tasks_df = openml.tasks.list_tasks(data_id=dataset_id, output_format="dataframe")
        if tasks_df.empty:
            print(f"   -> No tasks found for dataset_id: {dataset_id}")
            return []

        task_ids = tasks_df["tid"].tolist()
        evaluations = openml.evaluations.list_evaluations(
            function=function,
            tasks=task_ids,
            sort_order="desc",
            output_format="dataframe",
        )

        if evaluations.empty:
            print(f"   -> No evaluations found for any tasks on dataset_id: {dataset_id}")
            return []

        top_evaluations = evaluations.drop_duplicates(subset=["flow_id"]).head(n_setups)

        setups = []
        for setup_id in top_evaluations["setup_id"].unique():
            # CHANGE: Broaden the exception handling to catch any error with a
            # single problematic setup, making the loop more robust.
            try:
                setup = openml.setups.get_setup(setup_id)
                evaluation_row = top_evaluations[top_evaluations["setup_id"] == setup_id].iloc[0]

                # Check if parameters exist before trying to access them
                if setup.parameters is None:
                    print(f"   -> Skipping setup {setup_id}: No parameters found.")
                    continue

                params = {p.parameter_name: p.value for p in setup.parameters.values()}
                setups.append(
                    {
                        "flow_name": evaluation_row["flow_name"],
                        "performance": round(evaluation_row["value"], 4),
                        "hyperparameters": params,
                    }
                )
            except Exception as e:
                # This will now catch AttributeError, OpenMLServerException, and others.
                print(f"   -> Could not process setup {setup_id}. Reason: {e}")
                continue
        return setups

    def extract_suggested_config_space_parameters(self, dataset_name_in_openml: str) -> list[dict]:
        print(f"1. Searching for source dataset: '{dataset_name_in_openml}'...")
        source_dataset = self.get_source_dataset(dataset_name=dataset_name_in_openml)
        if source_dataset is None:
            print(f"❌ Error: No datasets found for '{dataset_name_in_openml}'.")
            return []

        source_id = int(source_dataset.name)
        print(f"   ✅ Found '{source_dataset['name']}' with ID: {source_id}")

        print(f"\n2. Finding the top 3 datasets similar to '{source_dataset['name']}'...")
        try:
            similar_datasets = self.find_similar_datasets(source_dataset_id=source_id, n_similar=3)
            print("   ✅ Found similar datasets:", similar_datasets["name"].tolist())
        except ValueError as e:
            print(f"❌ Error: {e}")
            return []

        print("\n3. Gathering top-performing setups from these similar datasets...")
        all_setups = []
        for dataset_id, dataset_row in similar_datasets.iterrows():
            print(f" -> Processing similar dataset: '{dataset_row['name']}' (ID: {dataset_id})")
            setups = self.get_top_setups_for_dataset(dataset_id, n_setups=3)
            if setups:
                all_setups.extend(setups)

        if not all_setups:
            print("❌ Error: Could not find any valid setups for the similar datasets.")
            return []

        print(f"\n   ✅ Gathered {len(all_setups)} total setups.")
        print("\n4. Cleaning and simplifying the final list...")
        final_hyperparameters = [setup["hyperparameters"] for setup in all_setups]
        unique_configs = list(OrderedDict((tuple(sorted(d.items())), d) for d in final_hyperparameters).values())
        print(f"   ✅ Final unique configurations found: {len(unique_configs)}")
        return unique_configs
