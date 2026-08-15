import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_FILE = Path(
    r"C:\Users\srira\Downloads\archive (8)\arxiv-metadata-oai-snapshot.json"
)

OUTPUT_FILE = Path("data/arxiv_cs_ai_ml.jsonl")

# Computer Science areas relevant to our chatbot
TARGET_CATEGORIES = {
    "cs.AI",   # Artificial Intelligence
    "cs.LG",   # Machine Learning
    "cs.CL",   # Natural Language Processing
    "cs.CV",   # Computer Vision
    "cs.IR",   # Information Retrieval
    "cs.NE",   # Neural and Evolutionary Computing
    "cs.RO",   # Robotics
}

# For our project, keep the first 50,000 matching papers.
MAX_PAPERS = 50000


# ============================================================
# CATEGORY PARSER
# ============================================================

def get_categories(category_text):
    if not category_text:
        return set()

    return set(category_text.split())


# ============================================================
# FILTER DATASET
# ============================================================

def filter_dataset():

    if not SOURCE_FILE.exists():

        print()
        print("=" * 60)
        print("ERROR: Dataset file not found")
        print("=" * 60)

        print(SOURCE_FILE)

        print()
        print("Check the SOURCE_FILE path.")
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_records = 0
    selected_papers = 0

    print()
    print("=" * 60)
    print("ARXIV DATASET FILTER")
    print("=" * 60)

    print(f"Source file: {SOURCE_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    print()
    print("Target Computer Science categories:")

    for category in sorted(TARGET_CATEGORIES):
        print(f"  {category}")

    print()
    print("Scanning the large arXiv dataset...")
    print("Please wait.")
    print("=" * 60)

    with SOURCE_FILE.open(
        "r",
        encoding="utf-8"
    ) as source:

        with OUTPUT_FILE.open(
            "w",
            encoding="utf-8"
        ) as output:

            for line in source:

                line = line.strip()

                if not line:
                    continue

                try:
                    paper = json.loads(line)

                except json.JSONDecodeError:
                    continue

                total_records += 1

                categories = get_categories(
                    paper.get("categories", "")
                )

                # Check whether the paper belongs
                # to one of our target CS categories.
                if not categories.intersection(
                    TARGET_CATEGORIES
                ):
                    continue

                cleaned_paper = {
                    "id": paper.get("id", ""),
                    "title": paper.get("title", "").strip(),
                    "abstract": paper.get("abstract", "").strip(),
                    "authors": paper.get("authors", ""),
                    "categories": paper.get("categories", ""),
                    "update_date": paper.get(
                        "update_date",
                        ""
                    )
                }

                output.write(
                    json.dumps(
                        cleaned_paper,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                selected_papers += 1

                if selected_papers % 5000 == 0:

                    print(
                        f"Selected papers: "
                        f"{selected_papers:,}"
                    )

                if selected_papers >= MAX_PAPERS:

                    print()
                    print(
                        f"Reached limit of "
                        f"{MAX_PAPERS:,} papers."
                    )

                    break

    print()
    print("=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)

    print(
        f"Records scanned: "
        f"{total_records:,}"
    )

    print(
        f"Papers selected: "
        f"{selected_papers:,}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    filter_dataset()