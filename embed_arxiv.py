import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/arxiv_chunks.jsonl")
OUTPUT_FILE = Path("data/arxiv_embeddings.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Number of chunks embedded at once
BATCH_SIZE = 64


# ============================================================
# HELPER: FIND TEXT FIELD
# ============================================================

def get_chunk_text(chunk):
    """
    Find the text/content field used by arxiv_chunks.jsonl.

    Supports several possible field names so the script
    does not depend on one exact schema.
    """

    possible_fields = [
        "text",
        "chunk_text",
        "content",
        "chunk",
        "text_chunk",
        "body"
    ]

    for field in possible_fields:
        value = chunk.get(field)

        if isinstance(value, str) and value.strip():
            return value

    # Last fallback: combine title + abstract
    title = chunk.get("title", "")
    abstract = chunk.get("abstract", "")

    if title or abstract:
        return f"{title}\n\n{abstract}"

    raise ValueError(
        f"Could not find text content in chunk.\n"
        f"Available fields: {list(chunk.keys())}"
    )


# ============================================================
# START
# ============================================================

print("=" * 60)
print("ARXIV EMBEDDING PIPELINE")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# READ CHUNKS
# ============================================================

print("\nLoading chunks...")

chunks = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():

            chunk = json.loads(line)

            chunks.append(chunk)


print(f"Chunks loaded: {len(chunks):,}")


# ============================================================
# CHECK FIRST CHUNK
# ============================================================

print("\nChecking chunk structure...")

first_chunk = chunks[0]

print("Available fields:")

for field in first_chunk.keys():
    print(f"  - {field}")


# Find the actual text field
sample_text = get_chunk_text(first_chunk)

print("\nText field detected successfully.")
print(f"Sample text length: {len(sample_text)} characters")


# ============================================================
# CREATE OUTPUT FILE
# ============================================================

print("\nStarting embedding generation...")

# Remove old incomplete output if it exists
if OUTPUT_FILE.exists():

    print("Removing existing embedding file...")

    OUTPUT_FILE.unlink()


# ============================================================
# PROCESS IN BATCHES
# ============================================================

total_chunks = len(chunks)

processed = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:

    for start in tqdm(
        range(0, total_chunks, BATCH_SIZE),
        desc="Embedding batches"
    ):

        end = min(start + BATCH_SIZE, total_chunks)

        batch = chunks[start:end]


        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        texts = [
            get_chunk_text(chunk)
            for chunk in batch
        ]


        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True
        )


        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        for chunk, embedding in zip(batch, embeddings):

            record = {
                "paper_id": chunk.get("paper_id"),
                "title": chunk.get("title"),
                "authors": chunk.get("authors"),
                "categories": chunk.get("categories"),
                "chunk_id": chunk.get("chunk_id"),

                # Preserve the original text
                "text": get_chunk_text(chunk),

                # 768-dimensional BGE vector
                "embedding": embedding.tolist()
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


        processed = end


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDING PIPELINE COMPLETE")
print("=" * 60)

print(f"Chunks processed: {processed:,}")
print("Embedding model: BAAI/bge-base-en-v1.5")
print("Embedding dimensions: 768")
print(f"Saved to: {OUTPUT_FILE}")

print("=" * 60)