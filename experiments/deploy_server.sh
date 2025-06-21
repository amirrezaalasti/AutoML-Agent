#!/bin/bash

# AutoML Agent - Server Deployment Script
# This script runs batch processing on a server environment

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/batch_results"
LOG_DIR="$PROJECT_ROOT/logs/server"

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "🤖 AutoML Agent - Server Deployment"
echo "=================================="
echo "Project Root: $PROJECT_ROOT"
echo "Output Directory: $OUTPUT_DIR"
echo "Log Directory: $LOG_DIR"
echo "Timestamp: $(date)"
echo ""

# Check if Python and required packages are available
echo "Checking dependencies..."
python3 -c "import pandas, sklearn, streamlit" 2>/dev/null || {
    echo "❌ Required Python packages not found. Please install requirements:"
    echo "   pip install -r requirements.txt"
    exit 1
}

# Check if API key is configured
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY environment variable not set"
    echo "   Please set your Google API key:"
    echo "   export GOOGLE_API_KEY='your-api-key-here'"
    exit 1
fi

echo "✅ Dependencies check passed"
echo ""

# Run the batch processing
echo "Starting batch processing..."
cd "$PROJECT_ROOT"

# Run with nohup to keep it running even if terminal disconnects
nohup python3 "$SCRIPT_DIR/run_batch_server.py" > "$LOG_DIR/batch_stdout.log" 2> "$LOG_DIR/batch_stderr.log" &

# Get the process ID
BATCH_PID=$!
echo "Batch processing started with PID: $BATCH_PID"
echo "Logs will be written to:"
echo "  - stdout: $LOG_DIR/batch_stdout.log"
echo "  - stderr: $LOG_DIR/batch_stderr.log"
echo "  - individual dataset logs: $LOG_DIR/*_server.log"
echo ""

# Save PID to file for later reference
echo $BATCH_PID > "$LOG_DIR/batch.pid"

echo "To monitor progress:"
echo "  tail -f $LOG_DIR/batch_stdout.log"
echo ""
echo "To check if process is still running:"
echo "  ps -p $BATCH_PID"
echo ""
echo "To stop the process:"
echo "  kill $BATCH_PID"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"
echo ""

# Wait a moment and show initial output
sleep 2
echo "Initial output:"
tail -n 10 "$LOG_DIR/batch_stdout.log" 2>/dev/null || echo "No output yet..."

echo ""
echo "✅ Deployment completed successfully!"
echo "Process is running in background with PID: $BATCH_PID" 