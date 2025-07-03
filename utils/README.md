# Utils Package

This package contains utility functions for the AutoML Agent project, organized into logical modules for better maintainability and code organization.

## Package Structure

```
utils/
├── __init__.py              # Main package file with all exports
├── constants.py             # Constants and configuration values
├── types.py                 # Type definitions and custom exceptions
├── data_processing.py       # Dataset validation, formatting, and splitting
├── dataset_description.py   # Dataset description generation functions
├── file_operations.py       # File handling and code operations
└── README.md               # This file
```

## Modules

### `constants.py`
Contains all constant values, default parameters, and mappings used throughout the system:
- Default test sizes, random states, SMAC parameters
- Supported dataset types
- Dataset task type mappings

### `types.py`
Contains type definitions and custom exceptions:
- `DatasetInfo` dataclass for holding dataset information
- `DatasetError` and `ValidationError` custom exceptions

### `data_processing.py`
Functions for dataset handling:
- `validate_dataset()` - Validate dataset format and structure
- `validate_dataset_type()` - Validate dataset type
- `get_dataset_info()` - Extract basic information from dataset
- `format_dataset()` - Format and preprocess dataset
- `split_dataset_kfold()` - Split dataset using K-fold cross-validation

### `dataset_description.py`
Functions for generating dataset descriptions:
- `describe_dataset()` - Main function for dataset description
- `describe_tabular_dataset()` - Tabular dataset descriptions
- `describe_time_series_dataset()` - Time series dataset descriptions
- `describe_image_dataset()` - Image dataset descriptions
- `describe_categorical_dataset()` - Categorical dataset descriptions
- `describe_text_dataset()` - Text dataset descriptions
- `add_general_description()` - Add general dataset info
- `generate_dataset_task_types()` - Generate task types for dataset

### `file_operations.py`
Functions for file handling:
- `extract_code_block()` - Extract Python code from markdown
- `save_code_to_file()` - Save generated code to file
- `convert_dataset_to_csv()` - Convert dataset to CSV format

## Usage

Import the utilities you need:

```python
# Import specific functions
from utils import format_dataset, describe_dataset

# Import all utilities
from utils import *

# Import from specific modules
from utils.data_processing import validate_dataset
from utils.constants import SUPPORTED_DATASET_TYPES
```

## Migration from scripts/utils.py

This package replaces the monolithic `scripts/utils.py` file. All functions have been preserved and organized into logical modules. The main `__init__.py` file exports all functions, so existing code should continue to work with minimal changes to import statements.

### Import Changes Required

Old:
```python
from scripts.utils import format_dataset
```

New:
```python
from utils import format_dataset
``` 