import requests
from bs4 import BeautifulSoup
import os
import json
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class DocumentCollector:
    def __init__(self, max_workers: int = 5, timeout: int = 10):
        self.session = requests.Session()
        self.timeout = timeout
        self.max_workers = max_workers

        self.base_urls: Dict[str, str] = {
            "smac": "https://automl.github.io/SMAC3/latest/",
            "configspace": "https://automl.github.io/ConfigSpace/latest/",
            "pytorch": "https://pytorch.org/docs/stable/",
            "tensorflow": "https://www.tensorflow.org/api_docs/",
            "sklearn": "https://scikit-learn.org/stable/",
        }

        self.sections = {
            "smac": [
                "3_getting_started/",
                "4_minimal_example.html",
                "examples/index.html",
                "advanced_usage/1_components/",
                "advanced_usage/2_multi_fidelity/",
                "advanced_usage/3_logging/",
            ],
            "configspace": [
                "quickstart/",
                "guide/",
                "api/ConfigSpace/configuration_space/",
                "api/ConfigSpace/configuration/",
            ],
            "pytorch": [
                "notes/training-tricks.html",
                "notes/recipes.html",
                "notes/multiprocessing.html",
                "notes/amp_examples.html",
                "notes/optimization.html",
            ],
            "tensorflow": [
                "training_with_built_in_methods",
                "customizing_what_happens_in_fit",
                "writing_a_training_loop_from_scratch",
                "distributed_training",
                "mixed_precision",
            ],
            "sklearn": [
                "modules/cross_validation.html",
                "modules/grid_search.html",
                "modules/model_evaluation.html",
                "modules/model_persistence.html",
                "common_pitfalls.html",
            ],
        }

    def extract_main_content(self, tool: str, soup: BeautifulSoup) -> Optional[str]:
        """
        Extracts the main content from a BeautifulSoup-parsed page based on tool.
        Falls back to <article> if specific selectors are not found.
        """
        selectors = {
            "pytorch": [("div", {"class": "section"}), ("article", {})],
            "tensorflow": [("main", {}), ("article", {})],
            "sklearn": [
                ("div", {"class": "section"}),
                ("div", {"class": "body"}),
                ("article", {}),
            ],
            # default for smac and configspace
            "default": [("article", {}), ("div", {"class": "body"})],
        }
        candidates = selectors.get(tool, selectors["default"])
        for tag, attrs in candidates:
            content = soup.find(tag, attrs)
            if content and content.get_text(strip=True):
                return content.get_text(separator="\n", strip=True)
        return None

    def fetch_section(self, tool: str, section: str) -> Optional[Dict[str, str]]:
        base_url = self.base_urls.get(tool, "")
        url = base_url + section
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as ex:
            logger.warning(f"Failed to fetch {url}: {ex}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        content = self.extract_main_content(tool, soup)
        if not content:
            logger.warning(f"No main content found for {url}")
            return None

        title_tag = soup.find(["h1", "title"])
        title = title_tag.get_text(strip=True) if title_tag else section
        return {"title": title, "content": content, "url": url}

    def collect_documentation(self) -> Dict[str, List[Dict[str, str]]]:
        """Collect documentation from all configured sources concurrently."""
        documents: Dict[str, List[Dict[str, str]]] = {tool: [] for tool in self.sections}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for tool, secs in self.sections.items():
                for sec in secs:
                    futures.append(executor.submit(self.fetch_section, tool, sec))
            for future in as_completed(futures):
                result = future.result()
                if result:
                    # Determine which tool it came from by URL
                    for tool in self.base_urls:
                        if result["url"].startswith(self.base_urls[tool]):
                            documents[tool].append(result)
                            break
        return documents

    def save_documents(
        self,
        documents: Dict[str, List[Dict[str, str]]],
        output_dir: str = "collected_docs",
    ):
        """Save collected documents to JSON files in the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        for category, docs in documents.items():
            output_file = os.path.join(output_dir, f"{category}_docs.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(docs, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(docs)} docs for '{category}' to {output_file}")


def main():
    collector = DocumentCollector(max_workers=8, timeout=15)
    logger.info("Starting document collection...")
    docs = collector.collect_documentation()
    collector.save_documents(docs)
    logger.info("Document collection completed.")


if __name__ == "__main__":
    main()
