from scripts.utils import load_image_dataset


image_datasets = [
    41983
]
for dataset_id in image_datasets:
    X, y = load_image_dataset(dataset_origin="openml", dataset_id=str(dataset_id), overwrite=True)
    print(X.head())
    print(y.head())
    break