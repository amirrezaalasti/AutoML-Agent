from typing import Any


class CLIUI:
    """Simple CLI UI agent for batch processing."""

    def __init__(self):
        self.current_dataset = None

    def subheader(self, text: str):
        """Display a subheader."""
        print(f"\n{'='*50}")
        print(f" {text}")
        print(f"{'='*50}")

    def write(self, text: Any):
        """Display text."""
        print(f"  {text}")

    def code(self, code: str, language: str = "python"):
        """Display code."""
        print(f"  [Code - {language}]:")
        print(f"  {code[:200]}..." if len(code) > 200 else f"  {code}")

    def success(self, message: str):
        """Display success message."""
        print(f"  ✅ {message}")

    def error(self, message: str):
        """Display error message."""
        print(f"  ❌ {message}")

    def warning(self, message: str):
        """Display warning message."""
        print(f"  ⚠️  {message}")

    def info(self, message: str):
        """Display info message."""
        print(f"  ℹ️  {message}")
