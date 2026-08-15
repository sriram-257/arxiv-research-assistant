import json
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/arxiv_cs_ai_ml.jsonl")
OUTPUT_FILE = Path("data/arxiv_chunks.jsonl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ============================================================
# CHUNKING FUNCTION
# ============================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# PROCESS ARXIV DATASET
# ============================================================

def process_dataset():

    print("Starting chunking process...")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    total_papers = 0
    total_chunks = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:

            paper = json.loads(line)

            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "").strip()
            categories = paper.get("categories", "")
            paper_id = paper.get("id", str(total_papers))

            # Combine title and abstract
            text = f"Title: {title}\n\nAbstract: {abstract}"

            chunks = create_chunks(text)

            for chunk_number, chunk in enumerate(chunks):

                record = {
                    "id": f"{paper_id}_chunk_{chunk_number}",
                    "paper_id": paper_id,
                    "title": title,
                    "categories": categories,
                    "chunk": chunk
                }

                outfile.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total_chunks += 1

            total_papers += 1

            if total_papers % 5000 == 0:
                print(
                    f"Papers processed: {total_papers:,} | "
                    f"Chunks created: {total_chunks:,}"
                )

    print("\n========================================")
    print("CHUNKING COMPLETE")
    print("========================================")
    print(f"Papers processed: {total_papers:,}")
    print(f"Chunks created: {total_chunks:,}")
    print(f"Saved to: {OUTPUT_FILE}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    process_dataset()