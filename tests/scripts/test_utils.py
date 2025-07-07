from scripts.utils import load_dataset, load_image_dataset

def test_load_dataset():
    X, y = load_dataset(dataset_origin="openml", dataset_id=31)
    assert X is not None
    assert y is not None
    assert len(X) == len(y)
    assert len(X) > 0
    assert len(y) > 0
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0
    assert X.shape[0] > 0

def test_load_image_dataset():
    X, y = load_image_dataset(dataset_origin="openml", dataset_id="41983", overwrite=True   )
    X_loaded, y_loaded = load_image_dataset(dataset_origin="openml", dataset_id="41983", overwrite=False)
    assert X is not None
    assert y is not None
    assert X_loaded is not None
    assert y_loaded is not None
    assert X.equals(X_loaded)
    assert y.equals(y_loaded)