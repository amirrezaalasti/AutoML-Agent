#!/usr/bin/env python3
"""
Batch processing monitor for server deployment.

This script monitors the status of batch processing and provides
real-time updates on progress and results.
"""

import os
import sys
import time
import json
import glob
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def check_process_status():
    """Check if the batch process is still running."""
    project_root = get_project_root()
    pid_file = project_root / "logs" / "server" / "batch.pid"

    if not pid_file.exists():
        return None, "No PID file found"

    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())

        # Check if process is running
        try:
            os.kill(pid, 0)  # This will raise an error if process doesn't exist
            return pid, "running"
        except OSError:
            return pid, "stopped"
    except Exception as e:
        return None, f"Error reading PID file: {e}"


def get_latest_logs():
    """Get the latest log entries."""
    project_root = get_project_root()
    log_dir = project_root / "logs" / "server"

    logs = []

    # Get main batch log
    batch_log = log_dir / "batch_stdout.log"
    if batch_log.exists():
        with open(batch_log, "r") as f:
            lines = f.readlines()
            logs.append(("Main Batch Log", lines[-20:]))  # Last 20 lines

    # Get individual dataset logs
    dataset_logs = glob.glob(str(log_dir / "*_server.log"))
    for log_file in dataset_logs:
        dataset_name = Path(log_file).stem.replace("_server", "")
        with open(log_file, "r") as f:
            lines = f.readlines()
            logs.append((f"Dataset: {dataset_name}", lines[-10:]))  # Last 10 lines

    return logs


def get_results_summary():
    """Get a summary of results."""
    project_root = get_project_root()
    output_dir = project_root / "batch_results"

    # Look for latest results
    csv_files = glob.glob(str(output_dir / "batch_results_*.csv"))
    json_files = glob.glob(str(output_dir / "batch_summary_*.json"))

    if not csv_files and not json_files:
        return "No results found yet"

    # Get latest files
    latest_csv = max(csv_files, key=os.path.getctime) if csv_files else None
    latest_json = max(json_files, key=os.path.getctime) if json_files else None

    summary = {}

    if latest_json:
        try:
            with open(latest_json, "r") as f:
                data = json.load(f)
                summary.update(
                    {
                        "total_datasets": data.get("total_datasets", 0),
                        "successful": data.get("successful", 0),
                        "failed": data.get("failed", 0),
                        "total_duration": data.get("total_duration_seconds", 0),
                        "timestamp": data.get("timestamp", "Unknown"),
                    }
                )
        except Exception as e:
            summary["error"] = f"Error reading JSON: {e}"

    if latest_csv:
        summary["csv_file"] = os.path.basename(latest_csv)

    return summary


def monitor_continuous():
    """Monitor continuously with updates."""
    print("🤖 AutoML Agent - Batch Processing Monitor")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print("")

    try:
        while True:
            # Clear screen (works on most terminals)
            os.system("clear" if os.name == "posix" else "cls")

            print("🤖 AutoML Agent - Batch Processing Monitor")
            print("=" * 50)
            print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("")

            # Check process status
            pid, status = check_process_status()
            if pid:
                print(f"📊 Process Status: {status.upper()} (PID: {pid})")
            else:
                print(f"📊 Process Status: {status}")
            print("")

            # Get results summary
            summary = get_results_summary()
            if isinstance(summary, dict):
                print("📈 Results Summary:")
                print(f"  Total Datasets: {summary.get('total_datasets', 'N/A')}")
                print(f"  Successful: {summary.get('successful', 'N/A')}")
                print(f"  Failed: {summary.get('failed', 'N/A')}")
                if summary.get("total_duration"):
                    print(f"  Total Duration: {summary['total_duration']:.1f}s")
                print(f"  Timestamp: {summary.get('timestamp', 'N/A')}")
                if "csv_file" in summary:
                    print(f"  Results File: {summary['csv_file']}")
            else:
                print(f"📈 Results: {summary}")
            print("")

            # Get latest logs
            logs = get_latest_logs()
            if logs:
                print("📝 Latest Logs:")
                for log_name, lines in logs[-3:]:  # Show last 3 log sources
                    print(f"  {log_name}:")
                    for line in lines[-3:]:  # Show last 3 lines
                        print(f"    {line.rstrip()}")
                    print("")

            # Wait before next update
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")


def show_status_once():
    """Show status once and exit."""
    print("🤖 AutoML Agent - Batch Processing Status")
    print("=" * 50)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # Check process status
    pid, status = check_process_status()
    if pid:
        print(f"📊 Process Status: {status.upper()} (PID: {pid})")
    else:
        print(f"📊 Process Status: {status}")
    print("")

    # Get results summary
    summary = get_results_summary()
    if isinstance(summary, dict):
        print("📈 Results Summary:")
        print(f"  Total Datasets: {summary.get('total_datasets', 'N/A')}")
        print(f"  Successful: {summary.get('successful', 'N/A')}")
        print(f"  Failed: {summary.get('failed', 'N/A')}")
        if summary.get("total_duration"):
            print(f"  Total Duration: {summary['total_duration']:.1f}s")
        print(f"  Timestamp: {summary.get('timestamp', 'N/A')}")
        if "csv_file" in summary:
            print(f"  Results File: {summary['csv_file']}")
    else:
        print(f"📈 Results: {summary}")
    print("")

    # Show recent logs
    logs = get_latest_logs()
    if logs:
        print("📝 Recent Activity:")
        for log_name, lines in logs[-2:]:  # Show last 2 log sources
            print(f"  {log_name}:")
            for line in lines[-2:]:  # Show last 2 lines
                print(f"    {line.rstrip()}")
            print("")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor AutoML Agent batch processing")
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Monitor continuously with updates",
    )
    parser.add_argument("--status", "-s", action="store_true", help="Show status once and exit")

    args = parser.parse_args()

    if args.continuous:
        monitor_continuous()
    else:
        show_status_once()
