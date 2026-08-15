import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_FILE = Path("data/arxiv_faiss.index")
METADATA_FILE = Path("data/arxiv_metadata.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("\nLoading FAISS index...")

index = faiss.read_index(str(INDEX_FILE))

print(f"Vectors in index: {index.ntotal:,}")
print(f"Dimensions: {index.d}")


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata...")

metadata = []

with open(METADATA_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():

            metadata.append(json.loads(line))


print(f"Metadata records: {len(metadata):,}")


# ============================================================
# USER QUERY
# ============================================================

query = input("\nEnter your research question: ")

print("\nSearching...")


# ============================================================
# CONVERT QUESTION → EMBEDDING
# ============================================================

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

# FAISS expects float32
query_embedding = query_embedding.astype("float32")


# ============================================================
# SEARCH FAISS
# ============================================================

scores, indices = index.search(
    query_embedding,
    TOP_K
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOP RESEARCH RESULTS")
print("=" * 70)

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    if idx < 0:
        continue

    result = metadata[idx]

    print(f"\n#{rank}")
    print("-" * 70)

    print(f"Similarity: {score:.4f}")

    print(f"Paper ID: {result.get('paper_id')}")

    print(f"Title: {result.get('title')}")

    print(f"Categories: {result.get('categories')}")

    print(f"\nText:")
    print(result.get("text"))

print("\n" + "=" * 70)