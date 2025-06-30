import requests
from bs4 import BeautifulSoup
import os
import json
import logging
import glob
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
            "smac2": "https://automl.github.io/SMAC3/v2.0.0/",
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
            "smac2": [
                "api/smac.facade.blackbox_facade.html",
                "api/smac.facade.abstract_facade.html",
                "api/smac.facade.algorithm_configuration_facade.html",
                "api/smac.facade.hyperband_facade.html",
                "api/smac.facade.hyperparameter_optimization_facade.html",
                "api/smac.facade.multi_fidelity_facade.html",
                "api/smac.facade.random_facade.html",
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
            "smac2": [
                # Primary content selectors for SMAC v2.0.0 documentation
                ("section", {"id": "interfaces"}),  # Main interfaces section
                ("section", {"id": "classes"}),  # Classes section
                ("div", {"class": "section"}),  # General section wrapper
                ("article", {}),  # Article wrapper
                ("div", {"class": "body"}),  # Body content
                ("main", {}),  # Main content area
            ],
            "smac": [
                ("article", {}),
                ("div", {"class": "body"}),
                ("div", {"class": "section"}),
            ],
            "configspace": [
                ("article", {}),
                ("div", {"class": "body"}),
                ("div", {"class": "section"}),
            ],
        }

        # Get candidates for this tool, fall back to default
        candidates = selectors.get(tool, [("article", {}), ("div", {"class": "body"})])

        logger.debug(f"Trying to extract content from {tool} using {len(candidates)} selectors")

        for i, (tag, attrs) in enumerate(candidates):
            logger.debug(f"Trying selector {i+1}: {tag} with {attrs}")
            content = soup.find(tag, attrs)
            if content:
                text = content.get_text(separator="\n", strip=True)
                if text and len(text.strip()) > 50:  # Ensure we have substantial content
                    logger.debug(f"Successfully extracted {len(text)} characters using selector {i+1}")
                    return text
                else:
                    logger.debug(f"Selector {i+1} found element but content too short: {len(text) if text else 0} chars")
            else:
                logger.debug(f"Selector {i+1} found no matching elements")

        # Last resort: try to get any meaningful content
        logger.debug("Trying fallback content extraction...")

        # Try to find any substantial text content
        all_text = soup.get_text(separator="\n", strip=True)
        if all_text and len(all_text.strip()) > 100:
            logger.debug(f"Using fallback extraction with {len(all_text)} characters")
            return all_text

        logger.warning(f"No substantial content found for {tool}")
        return None

    def fetch_section(self, tool: str, section: str) -> Optional[Dict[str, str]]:
        base_url = self.base_urls.get(tool, "")
        url = base_url + section

        logger.info(f"Fetching {tool}: {url}")

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            logger.debug(f"Successfully fetched {url} - Status: {resp.status_code}, Content-Length: {len(resp.text)}")
        except requests.RequestException as ex:
            logger.warning(f"Failed to fetch {url}: {ex}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Debug: Log some basic info about the page structure
        logger.debug(f"Page title: {soup.title.string if soup.title else 'No title'}")
        logger.debug(f"Found {len(soup.find_all('section'))} section tags")
        logger.debug(f"Found {len(soup.find_all('article'))} article tags")
        logger.debug(f"Found {len(soup.find_all('div', class_='section'))} div.section tags")

        content = self.extract_main_content(tool, soup)
        if not content:
            logger.warning(f"No main content found for {url}")
            # Debug: Save the HTML for inspection
            debug_file = f"debug_{tool}_{section.replace('/', '_').replace('.html', '')}.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(resp.text)
            logger.info(f"Saved HTML for debugging to {debug_file}")
            return None

        title_tag = soup.find(["h1", "title"])
        title = title_tag.get_text(strip=True) if title_tag else section

        logger.info(f"Successfully extracted content from {url} - Title: {title}, Content length: {len(content)}")
        return {"title": title, "content": content, "url": url}

    def test_single_url(self, tool: str, section: str) -> None:
        """Test a single URL for debugging purposes."""
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)

        print(f"\n=== Testing single URL ===")
        print(f"Tool: {tool}")
        print(f"Section: {section}")

        result = self.fetch_section(tool, section)
        if result:
            print(f"SUCCESS: Retrieved content")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Content length: {len(result['content'])}")
            print(f"Content preview (first 500 chars):")
            print("-" * 50)
            print(result["content"][:500])
            print("-" * 50)
        else:
            print("FAILED: No content retrieved")

        # Clean up debug files
        debug_files = glob.glob("debug_*.html")
        if debug_files:
            print(f"\nDebug files created: {', '.join(debug_files)}")
            print("You can inspect these HTML files to see the page structure")

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
    import sys

    # Set up logging to show DEBUG messages
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Show INFO and above in console
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    collector = DocumentCollector(max_workers=8, timeout=15)

    # Check if we want to test a single URL
    if len(sys.argv) == 3 and sys.argv[1] == "test":
        # Usage: python DocumentCollector.py test smac2
        # This will test the first section of the specified tool
        tool = sys.argv[2]
        if tool in collector.sections:
            section = collector.sections[tool][0]  # Test first section
            collector.test_single_url(tool, section)
        else:
            print(f"Unknown tool: {tool}")
            print(f"Available tools: {list(collector.sections.keys())}")
        return

    # Check if we want to test a specific URL
    if len(sys.argv) == 4 and sys.argv[1] == "test":
        # Usage: python DocumentCollector.py test smac2 api/smac.facade.blackbox_facade.html
        tool = sys.argv[2]
        section = sys.argv[3]
        collector.test_single_url(tool, section)
        return

    # Normal collection process
    logger.info("Starting document collection...")
    docs = collector.collect_documentation()
    collector.save_documents(docs)
    logger.info("Document collection completed.")

    # Print summary
    print("\n=== Collection Summary ===")
    for tool, documents in docs.items():
        print(f"{tool}: {len(documents)} documents collected")
        if len(documents) == 0:
            print(f"  ⚠️  No documents collected for {tool}")
        else:
            for doc in documents:
                print(f"  ✅ {doc['title']} ({len(doc['content'])} chars)")


if __name__ == "__main__":
    main()
