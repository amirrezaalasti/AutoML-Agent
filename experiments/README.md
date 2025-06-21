# AutoML Agent - Batch Processing

This directory contains scripts for running multiple datasets through the AutoML Agent in batch mode.

## Overview

The batch processing system provides three main approaches:

1. **Server Mode** - Headless execution optimized for server deployment (Recommended for servers)
2. **Streamlit UI Mode** - Interactive web interface with real-time progress tracking
3. **Command Line Mode** - Simple CLI interface for automated/headless execution

## Files

- `run_batch_server.py` - **Server-optimized batch runner** (Recommended for servers)
- `deploy_server.sh` - Deployment script for server environments
- `monitor_batch.py` - Monitoring script for batch processing status
- `run_multiple_datasets.py` - Streamlit-based batch runner with interactive UI
- `run_batch_cli.py` - Command-line batch runner for automated execution
- `BatchUI.py` - Streamlit UI component for batch processing
- `README.md` - This documentation file

## Usage

### Option 1: Server Mode (Recommended for Server Deployment)

**Deploy and run on server:**

```bash
# Set your API key
export GOOGLE_API_KEY='your-api-key-here'

# Deploy and start batch processing
./experiments/deploy_server.sh
```

**Monitor progress:**

```bash
# Check status once
python experiments/monitor_batch.py

# Monitor continuously
python experiments/monitor_batch.py --continuous
```

**Features:**
- ✅ **No UI dependencies** - Runs completely headless
- ✅ **Background execution** - Continues running if terminal disconnects
- ✅ **Comprehensive logging** - All output saved to log files
- ✅ **Process management** - PID tracking and easy monitoring
- ✅ **Automatic result export** - CSV and JSON output
- ✅ **Error recovery** - Individual dataset failures don't stop the batch
- ✅ **Server optimized** - Minimal resource usage

### Option 2: Streamlit UI Mode (Interactive Use)

Run the interactive batch processing interface:

```bash
streamlit run experiments/run_multiple_datasets.py
```

**Features:**
- Real-time progress tracking
- Interactive dataset selection
- Detailed results visualization
- Code inspection for each dataset
- Export functionality
- Error handling and display

### Option 3: Command Line Mode (Simple Automation)

Run batch processing from the command line:

```bash
python experiments/run_batch_cli.py
```

**Features:**
- Simple command-line interface
- Progress reporting
- Automatic CSV export
- Suitable for automation and scripts
- No GUI dependencies

## Server Deployment Guide

### Prerequisites

1. **Set API Key:**
   ```bash
   export GOOGLE_API_KEY='your-google-api-key-here'
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make Script Executable:**
   ```bash
   chmod +x experiments/deploy_server.sh
   ```

### Deployment Steps

1. **Start Batch Processing:**
   ```bash
   ./experiments/deploy_server.sh
   ```

2. **Monitor Progress:**
   ```bash
   # Check status
   python experiments/monitor_batch.py
   
   # Continuous monitoring
   python experiments/monitor_batch.py --continuous
   ```

3. **Check Logs:**
   ```bash
   # Main batch log
   tail -f logs/server/batch_stdout.log
   
   # Individual dataset logs
   tail -f logs/server/iris_server.log
   tail -f logs/server/wine_server.log
   ```

4. **View Results:**
   ```bash
   # List result files
   ls -la batch_results/
   
   # View latest results
   cat batch_results/batch_summary_*.json
   ```

### Server Management

**Check if process is running:**
```bash
ps -p $(cat logs/server/batch.pid)
```

**Stop the process:**
```bash
kill $(cat logs/server/batch.pid)
```

**View real-time logs:**
```bash
tail -f logs/server/batch_stdout.log
```

## Dataset Configuration

All scripts use the same dataset configuration format:

```python
openml_datasets = [
    {
        "dataset_name": "iris",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
    {
        "dataset_name": "wine",
        "dataset_type": "tabular", 
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
    # Add more datasets...
]
```

## Supported Datasets

Currently supported datasets:
- `iris` - Tabular classification dataset
- `wine` - Tabular classification dataset  
- `breast_cancer` - Tabular classification dataset
- `fashion-mnist` - Image classification dataset (loading not implemented yet)

## Output Structure

### Server Mode Output
```
batch_results/
├── batch_results_1703123456.csv          # Detailed results
├── batch_summary_1703123456.json         # Summary statistics
└── batch_run_1703123456.log              # Main execution log

logs/server/
├── batch.pid                             # Process ID
├── batch_stdout.log                      # Main stdout
├── batch_stderr.log                      # Main stderr
├── iris_server.log                       # Iris dataset log
├── wine_server.log                       # Wine dataset log
└── breast_cancer_server.log              # Breast cancer dataset log
```

### Result Files

**CSV Results** include:
- Dataset information (name, type, task)
- Processing status and duration
- Performance metrics (accuracy scores)
- Generated code (truncated)
- Timestamps and error messages

**JSON Summary** includes:
- Overall statistics (total, successful, failed)
- Duration information
- Complete results array
- Timestamp

## Error Handling

All modes include comprehensive error handling:

- **Individual dataset failures** don't stop the entire batch
- **Error reporting** with full tracebacks
- **Status tracking** for each dataset
- **Graceful degradation** when datasets fail
- **Log preservation** for debugging

## Performance Considerations

- **Sequential processing** - datasets run one after another
- **Memory usage** increases with dataset size and number of datasets
- **API rate limits** may apply when using LLM services
- **Disk space** needed for generated code and experiment logs
- **Server mode** optimized for minimal resource usage

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **API key issues**: Check your Google API key configuration
3. **Dataset loading errors**: Verify dataset names and loading functions
4. **Memory issues**: Reduce batch size or dataset complexity
5. **Permission errors**: Ensure scripts are executable

### Debug Mode

For debugging, you can:

**Check logs:**
```bash
# Main logs
tail -f logs/server/batch_stdout.log
tail -f logs/server/batch_stderr.log

# Individual dataset logs
tail -f logs/server/*_server.log
```

**Monitor process:**
```bash
python experiments/monitor_batch.py --continuous
```

**Check results:**
```bash
ls -la batch_results/
cat batch_results/batch_summary_*.json
```

## Example Server Output

### Deployment
```bash
🤖 AutoML Agent - Server Deployment
==================================
Project Root: /path/to/project
Output Directory: /path/to/project/batch_results
Log Directory: /path/to/project/logs/server
Timestamp: 2024-01-15 10:30:00

Checking dependencies...
✅ Dependencies check passed

Starting batch processing...
Batch processing started with PID: 12345
Logs will be written to:
  - stdout: logs/server/batch_stdout.log
  - stderr: logs/server/batch_stderr.log
  - individual dataset logs: logs/server/*_server.log

To monitor progress:
  tail -f logs/server/batch_stdout.log

To check if process is still running:
  ps -p 12345

To stop the process:
  kill 12345

Results will be saved to: batch_results

✅ Deployment completed successfully!
Process is running in background with PID: 12345
```

### Monitoring
```bash
🤖 AutoML Agent - Batch Processing Monitor
==================================================
Last updated: 2024-01-15 10:35:00

📊 Process Status: RUNNING (PID: 12345)

📈 Results Summary:
  Total Datasets: 3
  Successful: 2
  Failed: 0
  Total Duration: 142.3s
  Timestamp: 2024-01-15 10:32:15
  Results File: batch_results_1703123456.csv

📝 Latest Logs:
  Main Batch Log:
    2024-01-15 10:34:45 - batch_server - INFO - ✅ wine: SUCCESS (45.2s)
    2024-01-15 10:34:50 - batch_server - INFO - Processing dataset 3/3: breast_cancer
    2024-01-15 10:34:50 - batch_server - INFO - Type: tabular, Task: classification
```

## Future Enhancements

Potential improvements:
- Parallel processing support
- More dataset types (time series, text, etc.)
- Advanced result visualization
- Integration with experiment tracking systems
- Custom metric definitions
- Automated hyperparameter tuning
- Docker containerization
- Kubernetes deployment support 