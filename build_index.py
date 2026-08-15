import json
import numpy as np
import faiss
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/arxiv_embeddings.jsonl")
INDEX_FILE = Path("data/arxiv_faiss.index")
METADATA_FILE = Path("data/arxiv_metadata.jsonl")

EMBEDDING_DIM = 384

# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("=" * 60)
print("ARXIV FAISS INDEX BUILDER")
print("=" * 60)

print("\nLoading embeddings...")

embeddings = []
metadata = []

with INPUT_FILE.open("r", encoding="utf-8") as f:

    for line in f:

        record = json.loads(line)

        # The embedding is stored with each chunk
        embedding = record["embedding"]

        embeddings.append(embedding)

        # Keep the information needed to display search results
        metadata.append({
            "id": record.get("id"),
            "paper_id": record.get("paper_id"),
            "title": record.get("title"),
            "authors": record.get("authors"),
            "categories": record.get("categories"),
            "text": record.get("text")
        })

        if len(embeddings) % 5000 == 0:
            print(f"Loaded: {len(embeddings):,}")

# ============================================================
# CONVERT TO NUMPY
# ============================================================

print("\nConverting embeddings to NumPy...")

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print(f"Embedding matrix shape: {embeddings.shape}")

# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing embeddings...")

faiss.normalize_L2(embeddings)

# ============================================================
# CREATE FAISS INDEX
# ============================================================

print("\nCreating FAISS index...")

index = faiss.IndexFlatIP(EMBEDDING_DIM)

index.add(embeddings)

print(f"Vectors indexed: {index.ntotal:,}")

# ============================================================
# SAVE FAISS INDEX
# ============================================================

print("\nSaving FAISS index...")

faiss.write_index(
    index,
    str(INDEX_FILE)
)

# ============================================================
# SAVE METADATA
# ============================================================

print("Saving metadata...")

with METADATA_FILE.open("w", encoding="utf-8") as f:

    for item in metadata:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            ) + "\n"
        )

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("FAISS INDEX BUILD COMPLETE")
print("=" * 60)

print(f"Vectors indexed: {index.ntotal:,}")
print(f"Dimensions: {EMBEDDING_DIM}")
print(f"Index saved to: {INDEX_FILE}")
print(f"Metadata saved to: {METADATA_FILE}")

print("=" * 60)