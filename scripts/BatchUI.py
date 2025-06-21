"""
Batch UI Agent for running multiple datasets.

This module provides a UI agent that can handle multiple dataset runs
and display results in a structured, organized manner.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class DatasetResult:
    """Data class to hold results for a single dataset run."""

    dataset_name: str
    dataset_type: str
    task_type: str
    status: str  # "success", "error", "running"
    start_time: datetime
    end_time: Optional[datetime] = None
    config_code: Optional[str] = None
    scenario_code: Optional[str] = None
    train_function_code: Optional[str] = None
    last_loss: Optional[float] = None
    default_train_accuracy: Optional[float] = None
    incumbent_train_accuracy: Optional[float] = None
    test_train_accuracy: Optional[float] = None
    incumbent_config: Optional[str] = None
    error_message: Optional[str] = None
    experiment_dir: Optional[str] = None


class BatchUI:
    """UI agent for handling multiple dataset runs with organized display."""

    def __init__(self):
        """Initialize the Batch UI agent."""
        self.results: List[DatasetResult] = []
        self.current_dataset: Optional[str] = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the main UI layout."""
        st.set_page_config(page_title="AutoML Agent - Batch Processing", page_icon="🤖", layout="wide")

        st.title("🤖 AutoML Agent - Batch Processing")
        st.markdown("---")

        # Create sidebar for navigation
        self.create_sidebar()

        # Create main content area
        self.create_main_content()

    def create_sidebar(self):
        """Create the sidebar with navigation and controls."""
        with st.sidebar:
            st.header("📊 Batch Controls")

            # Overall progress
            if self.results:
                completed = len([r for r in self.results if r.status in ["success", "error"]])
                total = len(self.results)
                st.metric("Progress", f"{completed}/{total}")

                # Progress bar
                progress = completed / total if total > 0 else 0
                st.progress(progress)

            # Dataset selection
            if self.results:
                st.subheader("📋 Dataset Results")
                for i, result in enumerate(self.results):
                    status_emoji = {
                        "success": "✅",
                        "error": "❌",
                        "running": "🔄",
                    }.get(result.status, "⏳")

                    if st.button(f"{status_emoji} {result.dataset_name}", key=f"dataset_{i}"):
                        self.current_dataset = result.dataset_name

            # Export results
            if self.results:
                st.subheader("💾 Export")
                if st.button("Export Results"):
                    self.export_results()

    def create_main_content(self):
        """Create the main content area."""
        # Summary section
        self.create_summary_section()

        # Current dataset details
        if self.current_dataset:
            self.create_dataset_details()
        else:
            self.create_welcome_section()

    def create_summary_section(self):
        """Create the summary section showing overall statistics."""
        st.subheader("📈 Batch Summary")

        if not self.results:
            st.info("No datasets processed yet. Start a batch run to see results.")
            return

        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total = len(self.results)
            st.metric("Total Datasets", total)

        with col2:
            successful = len([r for r in self.results if r.status == "success"])
            st.metric("Successful", successful)

        with col3:
            failed = len([r for r in self.results if r.status == "error"])
            st.metric("Failed", failed)

        with col4:
            running = len([r for r in self.results if r.status == "running"])
            st.metric("Running", running)

        # Create results table
        self.create_results_table()

    def create_results_table(self):
        """Create a table showing all dataset results."""
        if not self.results:
            return

        st.subheader("📊 Results Table")

        # Prepare data for table
        table_data = []
        for result in self.results:
            duration = None
            if result.end_time and result.start_time:
                duration = (result.end_time - result.start_time).total_seconds()

            table_data.append(
                {
                    "Dataset": result.dataset_name,
                    "Type": result.dataset_type,
                    "Task": result.task_type,
                    "Status": result.status,
                    "Duration (s)": f"{duration:.1f}" if duration else "N/A",
                    "Default Acc": (f"{result.default_train_accuracy:.4f}" if result.default_train_accuracy else "N/A"),
                    "Incumbent Acc": (f"{result.incumbent_train_accuracy:.4f}" if result.incumbent_train_accuracy else "N/A"),
                    "Test Acc": (f"{result.test_train_accuracy:.4f}" if result.test_train_accuracy else "N/A"),
                }
            )

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def create_dataset_details(self):
        """Create detailed view for selected dataset."""
        result = next((r for r in self.results if r.dataset_name == self.current_dataset), None)
        if not result:
            return

        st.subheader(f"🔍 Dataset Details: {result.dataset_name}")

        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Type", result.dataset_type)
        with col2:
            st.metric("Task", result.task_type)
        with col3:
            st.metric("Status", result.status)

        # Performance metrics
        if result.status == "success":
            st.subheader("📊 Performance Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                if result.default_train_accuracy is not None:
                    st.metric("Default Accuracy", f"{result.default_train_accuracy:.4f}")

            with col2:
                if result.incumbent_train_accuracy is not None:
                    st.metric("Incumbent Accuracy", f"{result.incumbent_train_accuracy:.4f}")

            with col3:
                if result.test_train_accuracy is not None:
                    st.metric("Test Accuracy", f"{result.test_train_accuracy:.4f}")

        # Error details
        if result.status == "error" and result.error_message:
            st.subheader("❌ Error Details")
            st.error(result.error_message)

        # Generated code (collapsible)
        if result.config_code or result.scenario_code or result.train_function_code:
            st.subheader("💻 Generated Code")

            if result.config_code:
                with st.expander("Configuration Space Code"):
                    st.code(result.config_code, language="python")

            if result.scenario_code:
                with st.expander("Scenario Code"):
                    st.code(result.scenario_code, language="python")

            if result.train_function_code:
                with st.expander("Training Function Code"):
                    st.code(result.train_function_code, language="python")

    def create_welcome_section(self):
        """Create welcome section when no dataset is selected."""
        st.subheader("🎯 Welcome to AutoML Agent Batch Processing")
        st.markdown(
            """
        This interface allows you to run multiple datasets in batch mode.

        **Features:**
        - 📊 Real-time progress tracking
        - 📈 Performance metrics comparison
        - 💻 Generated code inspection
        - 💾 Results export functionality

        **How to use:**
        1. Start a batch run with your dataset list
        2. Monitor progress in the sidebar
        3. Click on any dataset to view detailed results
        4. Export results when complete
        """
        )

    def export_results(self):
        """Export results to CSV."""
        if not self.results:
            st.warning("No results to export")
            return

        # Prepare export data
        export_data = []
        for result in self.results:
            duration = None
            if result.end_time and result.start_time:
                duration = (result.end_time - result.start_time).total_seconds()

            export_data.append(
                {
                    "dataset_name": result.dataset_name,
                    "dataset_type": result.dataset_type,
                    "task_type": result.task_type,
                    "status": result.status,
                    "start_time": result.start_time.isoformat(),
                    "end_time": (result.end_time.isoformat() if result.end_time else None),
                    "duration_seconds": duration,
                    "default_train_accuracy": result.default_train_accuracy,
                    "incumbent_train_accuracy": result.incumbent_train_accuracy,
                    "test_train_accuracy": result.test_train_accuracy,
                    "incumbent_config": result.incumbent_config,
                    "error_message": result.error_message,
                    "experiment_dir": result.experiment_dir,
                }
            )

        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"automl_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    # Methods for integration with AutoMLAgent
    def add_dataset(self, dataset_name: str, dataset_type: str, task_type: str) -> str:
        """Add a new dataset to the batch processing."""
        result = DatasetResult(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            task_type=task_type,
            status="running",
            start_time=datetime.now(),
        )
        self.results.append(result)
        return dataset_name

    def update_dataset_status(self, dataset_name: str, status: str, **kwargs):
        """Update the status and details of a dataset."""
        result = next((r for r in self.results if r.dataset_name == dataset_name), None)
        if result:
            result.status = status
            if status in ["success", "error"]:
                result.end_time = datetime.now()

            # Update other fields
            for key, value in kwargs.items():
                if hasattr(result, key):
                    setattr(result, key, value)

    def get_dataset_result(self, dataset_name: str) -> Optional[DatasetResult]:
        """Get the result for a specific dataset."""
        return next((r for r in self.results if r.dataset_name == dataset_name), None)

    # UI methods that mimic the original UI agent interface
    def subheader(self, text: str):
        """Display a subheader."""
        st.subheader(text)

    def write(self, text: Any):
        """Display text."""
        st.write(text)

    def code(self, code: str, language: str = "python"):
        """Display code."""
        st.code(code, language=language)

    def success(self, message: str):
        """Display success message."""
        st.success(message)

    def error(self, message: str):
        """Display error message."""
        st.error(message)

    def warning(self, message: str):
        """Display warning message."""
        st.warning(message)

    def info(self, message: str):
        """Display info message."""
        st.info(message)
