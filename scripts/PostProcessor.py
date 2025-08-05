"""
Post-processing module for analyzing and improving LLM-generated solutions
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd


class PostProcessor:
    """Handles post-processing analysis and improvement suggestions for LLM-generated solutions"""

    def __init__(self, ui_agent, logger):
        """
        Initialize the PostProcessor

        Args:
            ui_agent: UI agent for displaying results
            logger: Logger for recording analysis
        """
        self.ui_agent = ui_agent
        self.logger = logger
        self.baseline_accuracy: Optional[float] = None
        self.baseline_metrics: Optional[Dict] = None
        self.improvement_feedback: Optional[str] = None
        self.iteration_count = 0

    def set_baseline(self, baseline_accuracy: float, baseline_metrics: Dict):
        """Set the baseline performance metrics"""
        self.baseline_accuracy = baseline_accuracy
        self.baseline_metrics = baseline_metrics

    def analyze_performance(
        self,
        final_metrics: Dict,
        incumbent: Any,
        config_code: str,
        scenario_code: str,
        train_code: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Analyze the performance and generate improvement feedback if needed

        Args:
            final_metrics: Final evaluation metrics
            incumbent: Best configuration found by SMAC
            config_code: Generated configuration space code
            scenario_code: Generated scenario code
            train_code: Generated training function code

        Returns:
            Tuple of (baseline_beaten, improvement_feedback)
        """
        if self.baseline_accuracy is None:
            return True, None  # No baseline to compare against

        # Extract final accuracy
        final_accuracy = self._extract_accuracy(final_metrics)
        if final_accuracy is None:
            return True, None  # Could not extract accuracy

        # Check if baseline was beaten
        if final_accuracy >= self.baseline_accuracy:
            return True, None  # Baseline beaten, no improvement needed

        # Generate improvement feedback
        self.improvement_feedback = self._generate_improvement_feedback(
            final_accuracy,
            final_metrics,
            incumbent,
            config_code,
            scenario_code,
            train_code,
        )

        # Log the analysis
        self._log_analysis(final_accuracy, final_metrics, incumbent)

        # Display results to user
        self._display_analysis(final_accuracy)

        return False, self.improvement_feedback

    def _extract_accuracy(self, metrics: Dict) -> Optional[float]:
        """Extract accuracy from metrics dictionary"""
        if "accuracy" in metrics:
            return metrics["accuracy"]
        elif "balanced_accuracy" in metrics:
            return metrics["balanced_accuracy"]
        elif "f1" in metrics:
            return metrics["f1"]
        elif "loss" in metrics:
            return 1.0 - metrics["loss"]
        return None

    def _generate_improvement_feedback(
        self,
        final_accuracy: float,
        final_metrics: Dict,
        incumbent: Any,
        config_code: str,
        scenario_code: str,
        train_code: str,
    ) -> str:
        """
        Generate comprehensive improvement feedback

        Args:
            final_accuracy: Final accuracy achieved
            final_metrics: All final metrics
            incumbent: Best configuration found
            config_code: Current configuration space code
            scenario_code: Current scenario code
            train_code: Current training function code

        Returns:
            Detailed improvement feedback
        """
        self.iteration_count += 1
        accuracy_gap = self.baseline_accuracy - final_accuracy
        improvement_needed = accuracy_gap / self.baseline_accuracy * 100

        feedback = f"""
        ## **Iteration {self.iteration_count} - Baseline Performance Analysis**

        ### **Performance Summary:**
        - **Baseline Accuracy**: {self.baseline_accuracy:.4f} (AutoGluon)
        - **Generated Solution Accuracy**: {final_accuracy:.4f}
        - **Accuracy Gap**: {accuracy_gap:.4f} ({improvement_needed:.1f}% improvement needed)

        ### **Current Solution Analysis:**
        - **Best Configuration**: {incumbent}
        - **All Metrics**: {final_metrics}

        ### **Current Code Analysis:**

        #### **Configuration Space Issues:**
        - **Current Config**: The configuration space may be too limited for this dataset
        - **Suggestions**:
        - Add more algorithm families (XGBoost, LightGBM, CatBoost, Neural Networks)
        - Expand hyperparameter ranges for existing algorithms
        - Include ensemble methods and stacking approaches
        - Add feature engineering hyperparameters

        #### **Training Function Issues:**
        - **Current Implementation**: May lack sophisticated preprocessing and training strategies
        - **Suggestions**:
        - Implement comprehensive feature engineering
        - Add cross-validation and early stopping
        - Use more sophisticated algorithms
        - Improve data preprocessing pipeline

        #### **Scenario Configuration Issues:**
        - **Current Setup**: May not be optimal for the dataset characteristics
        - **Suggestions**:
        - Adjust time budget allocation
        - Optimize number of trials
        - Consider different SMAC facades
        - Fine-tune parallelization settings

        ### **Specific Improvement Actions:**

        #### **1. Configuration Space Enhancement:**
        - **Add Algorithm Diversity**: Include gradient boosting, deep learning, and ensemble methods
        - **Expand Hyperparameters**: Add more parameters for each algorithm
        - **Feature Engineering**: Include preprocessing and feature selection options
        - **Regularization**: Add regularization parameters for all algorithms

        #### **2. Training Function Improvements:**
        - **Feature Engineering**: Implement comprehensive preprocessing pipeline
        - **Cross-Validation**: Use proper validation strategies
        - **Early Stopping**: Prevent overfitting with early stopping
        - **Algorithm Selection**: Implement more sophisticated algorithm selection logic

        #### **3. Scenario Optimization:**
        - **Time Budget**: Allocate more time for complex algorithms
        - **Trial Count**: Increase number of trials for better exploration
        - **Parallelization**: Optimize worker configuration
        - **Facade Selection**: Consider different SMAC facades based on dataset

        ### **Target Goals:**
        - **Minimum Target**: Achieve at least {self.baseline_accuracy:.4f} accuracy
        - **Preferred Target**: Achieve {self.baseline_accuracy + 0.05:.4f} accuracy for significant improvement
        - **Success Criteria**: Beat AutoGluon baseline with robust, production-ready code

        ### **Previous Code for Reference:**
        ```
        Configuration Space:
        {config_code}

        Scenario:
        {scenario_code}

        Training Function:
        {train_code}
        ```
        """

        return feedback

    def _log_analysis(self, final_accuracy: float, final_metrics: Dict, incumbent: Any):
        """Log the performance analysis"""
        self.logger.log_response(
            f"Baseline performance analysis completed (Iteration {self.iteration_count})",
            {
                "component": "post_processing",
                "status": "baseline_failure",
                "iteration": self.iteration_count,
                "final_accuracy": final_accuracy,
                "baseline_accuracy": self.baseline_accuracy,
                "accuracy_gap": self.baseline_accuracy - final_accuracy,
                "improvement_feedback": self.improvement_feedback,
                "incumbent": str(incumbent),
                "final_metrics": final_metrics,
            },
        )

    def _display_analysis(self, final_accuracy: float):
        """Display analysis results to user"""
        self.ui_agent.subheader(f"Iteration {self.iteration_count} - Baseline Performance Analysis")
        self.ui_agent.warning(f"Generated solution ({final_accuracy:.4f}) did not beat baseline ({self.baseline_accuracy:.4f})")
        self.ui_agent.info(f"Improvement suggestions generated for iteration {self.iteration_count + 1}")

    def create_improvement_prompt(self, component_type: str) -> str:
        """
        Create an improvement prompt for the specified component type

        Args:
            component_type: Type of component to improve ('config', 'scenario', 'train_function')

        Returns:
            Improvement prompt string
        """
        if not self.improvement_feedback:
            return ""

        improvement_context = f"""
        ### **CRITICAL IMPROVEMENT CONTEXT - Iteration {self.iteration_count + 1}:**
        The previous generated solution achieved accuracy below the AutoGluon baseline of {self.baseline_accuracy:.4f}.

        ### **IMPROVEMENT REQUIREMENTS:**
        {self.improvement_feedback}

        ### **SPECIFIC INSTRUCTIONS FOR {component_type.upper()} IMPROVEMENT:**
        """

        if component_type == "config":
            improvement_context += """
            1. **MUST** expand algorithm diversity (XGBoost, LightGBM, CatBoost, Neural Networks)
            2. **MUST** add more hyperparameters for each algorithm
            3. **MUST** include feature engineering hyperparameters
            4. **MUST** add regularization parameters
            5. **MUST** consider ensemble methods and stacking
            6. **MUST** optimize hyperparameter ranges based on dataset characteristics
            """
        elif component_type == "scenario":
            improvement_context += """
            1. **MUST** optimize time budget allocation for complex algorithms
            2. **MUST** increase number of trials for better exploration
            3. **MUST** consider different SMAC facades
            4. **MUST** optimize parallelization settings
            5. **MUST** adjust resource constraints appropriately
            """
        elif component_type == "train_function":
            improvement_context += """
            1. **MUST** implement comprehensive feature engineering
            2. **MUST** use cross-validation and early stopping
            3. **MUST** implement more sophisticated algorithms
            4. **MUST** improve data preprocessing pipeline
            5. **MUST** target accuracy significantly above {self.baseline_accuracy:.4f}
            6. **MUST** be more aggressive in hyperparameter optimization
            7. **MUST** consider multiple algorithm families
            """

        improvement_context += f"""
        ### **SUCCESS CRITERIA:**
        - Achieve accuracy >= {self.baseline_accuracy:.4f}
        - Preferably achieve accuracy >= {self.baseline_accuracy + 0.05:.4f}
        - Implement robust, production-ready code
        - Beat the AutoGluon baseline performance
        """

        return improvement_context

    def get_iteration_info(self) -> Dict:
        """Get current iteration information"""
        return {
            "iteration_count": self.iteration_count,
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_metrics": self.baseline_metrics,
            "improvement_feedback": self.improvement_feedback,
        }

    def reset_iteration(self):
        """Reset iteration counter (useful for new datasets)"""
        self.iteration_count = 0
        self.improvement_feedback = None
