import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/arxiv_chunks.jsonl")

MODEL_NAME = "BAAI/bge-base-en-v1.5"

TEST_CHUNKS = 100


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("BGE EMBEDDING TEST")
print("=" * 60)

print()
print("Loading embedding model:")
print(MODEL_NAME)
print()

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")
print()


# ============================================================
# LOAD 100 CHUNKS
# ============================================================

documents = []

print(f"Reading first {TEST_CHUNKS} chunks...")

with INPUT_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    for _ in range(TEST_CHUNKS):

        line = file.readline()

        if not line:
            break

        paper = json.loads(line)

        documents.append(
            paper["chunk"]
        )


print(
    f"Loaded {len(documents)} chunks."
)

print()


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("Creating embeddings...")

embeddings = model.encode(
    documents,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ============================================================
# VERIFY
# ============================================================

print()
print("=" * 60)
print("EMBEDDING TEST COMPLETE")
print("=" * 60)

print(
    f"Number of embeddings: {len(embeddings)}"
)

print(
    f"Embedding dimensions: {embeddings.shape[1]}"
)

print(
    f"Embedding shape: {embeddings.shape}"
)

print()
print("Example vector:")
print(embeddings[0][:10])

print()
print("BGE embedding test successful!")
print("=" * 60)