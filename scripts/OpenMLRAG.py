import openml
from configs.api_keys import OPENML_API_KEY


class OpenMLRAG:
    def __init__(self):
        openml.config.apikey = OPENML_API_KEY

    def get_related_datasets(self, tag=None, dataset_name=None):
        """
        Get datasets related to a specific tag or dataset name.
        If dataset_name is provided, it will return datasets related to that specific dataset.
        If tag is provided, it will return datasets related to that tag.
        """
        datasets = openml.datasets.list_datasets(output_format="dataframe", status="active")
        if dataset_name:
            # choose the first dataset that matches the name
            datasets = datasets[datasets["name"].str.lower() == dataset_name.lower()]
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
        top_evaluations = evaluations.head(2)

        setup_parameters = []
        for evaluation in top_evaluations["setup_id"].unique():
            setup = openml.setups.get_setup(evaluation)
            setup_parameters.append(setup.parameters)

        return setup_parameters
