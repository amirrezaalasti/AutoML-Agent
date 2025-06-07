from scripts.AutoMLAppUI import AutoMLAppUI
import os


if __name__ == "__main__":
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    app_ui = AutoMLAppUI()
    app_ui.display()
