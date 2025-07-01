import openml
import pandas as pd
from autogluon.tabular import TabularPredictor

# 1. Download the "credit-g" dataset from OpenML
dataset = openml.datasets.get_dataset(31)
df, *_ = dataset.get_data()
print(df.head())

# 2. Identify the target column
target_col = dataset.default_target_attribute  # Should be 'class'

# 3. Optional: Split into train/test sets
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

# 4. Train AutoGluon
predictor = TabularPredictor(label=target_col).fit(train_data)

# 5. Evaluate on test data
performance = predictor.evaluate(test_data)
print(performance)