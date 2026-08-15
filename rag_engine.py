import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_FILE = Path("data/arxiv_faiss.index")
METADATA_FILE = Path("data/arxiv_metadata.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Retrieve a larger candidate pool first.
# These candidates are then reranked before selecting Top-3.
CANDIDATE_K = 50

# FINAL NUMBER OF PAPERS RETURNED TO THE APPLICATION
TOP_K = 3

# Relevant Computer Science categories
RELEVANT_CATEGORIES = {
    "cs.CL",   # Computation and Language / NLP
    "cs.AI",   # Artificial Intelligence
    "cs.LG",   # Machine Learning
    "cs.NE",   # Neural and Evolutionary Computing
}


# ============================================================
# DOMAIN TERMS
# ============================================================

NLP_TERMS = {
    "nlp",
    "natural language",
    "language model",
    "machine translation",
    "text",
    "question answering",
    "semantic",
    "token",
    "tokenization",
    "transformer",
    "transformers",
    "attention",
    "self-attention",
    "self attention",
    "bert",
    "gpt",
    "llm",
    "embedding",
    "sequence",
    "rnn",
    "recurrent neural network",
    "lstm",
    "gru",
}


AI_TERMS = {
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "neural networks",
    "classification",
    "regression",
    "prediction",
    "reinforcement learning",
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize(text: str) -> str:
    """
    Normalize text for matching and scoring.
    """

    text = str(text or "").lower()

    text = re.sub(
        r"[^a-z0-9\-\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokens(text: str) -> set[str]:
    """
    Convert text into a set of useful tokens.
    """

    return {
        token
        for token in normalize(text).split()
        if len(token) > 2
    }


def result_text(result: dict) -> str:
    """
    Get the main textual content from a metadata record.
    """

    return str(
        result.get("text")
        or result.get("chunk")
        or result.get("abstract")
        or ""
    )


def paper_id(result: dict) -> str:
    """
    Extract the ArXiv paper ID.
    """

    return str(
        result.get("paper_id")
        or result.get("id")
        or result.get("arxiv_id")
        or ""
    ).strip()


def paper_url(result: dict) -> str:
    """
    Generate the ArXiv paper URL.
    """

    pid = paper_id(result)

    if not pid:
        return ""

    return f"https://arxiv.org/abs/{pid}"


# ============================================================
# ARXIV RAG ENGINE
# ============================================================

class ArxivRAG:
    """
    FAISS-based retrieval engine for the existing
    50,000-record ArXiv knowledge base.
    """

    def __init__(self):

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        print("\nLoading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            f"Embedding model loaded: {MODEL_NAME}"
        )


        # ----------------------------------------------------
        # Load FAISS index
        # ----------------------------------------------------

        print("\nLoading FAISS index...")

        if not INDEX_FILE.exists():

            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_FILE}"
            )

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        print(
            f"Vectors in index: {self.index.ntotal:,}"
        )

        print(
            f"Index dimensions: {self.index.d}"
        )


        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        print("\nLoading metadata...")

        if not METADATA_FILE.exists():

            raise FileNotFoundError(
                f"Metadata file not found: {METADATA_FILE}"
            )

        self.metadata = []

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if line.strip():

                    self.metadata.append(
                        json.loads(line)
                    )


        print(
            f"Metadata records: {len(self.metadata):,}"
        )


        # ----------------------------------------------------
        # Verify FAISS / metadata consistency
        # ----------------------------------------------------

        if len(self.metadata) != self.index.ntotal:

            raise ValueError(
                "FAISS/metadata mismatch: "
                f"{self.index.ntotal} vectors vs "
                f"{len(self.metadata)} records."
            )


        # ----------------------------------------------------
        # Verify embedding dimensions
        # ----------------------------------------------------

        embedding_dimension = (
            self.model.get_embedding_dimension()
            if hasattr(
                self.model,
                "get_embedding_dimension"
            )
            else self.model.get_sentence_embedding_dimension()
        )


        if self.index.d != embedding_dimension:

            raise ValueError(
                "Embedding dimension mismatch: "
                f"index={self.index.d}, "
                f"model={embedding_dimension}"
            )


        print(
            "Index, metadata and embedding dimensions "
            "verified successfully."
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    def stats(self) -> dict:
        """
        Return information used by the Streamlit UI.
        """

        return {
            "records": self.index.ntotal,
            "dimensions": self.index.d,
            "embedding_model": MODEL_NAME,
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
        }


    # ========================================================
    # DOMAIN DETECTION
    # ========================================================

    def detect_domain(
        self,
        query: str
    ) -> str:

        q = normalize(query)

        if any(
            term in q
            for term in NLP_TERMS
        ):

            return "NLP"


        if any(
            term in q
            for term in AI_TERMS
        ):

            return "AI"


        return "GENERAL"


    # ========================================================
    # CATEGORY SCORE
    # ========================================================

    def _category_score(
        self,
        result: dict,
        domain: str
    ) -> float:

        categories = set(
            str(
                result.get(
                    "categories",
                    ""
                )
            ).split()
        )


        if domain == "NLP":

            score = (
                1.00
                if "cs.CL" in categories
                else 0.0
            )

            score += (
                0.60
                if "cs.AI" in categories
                else 0.0
            )

            score += (
                0.45
                if "cs.LG" in categories
                else 0.0
            )

            score += (
                0.25
                if "cs.NE" in categories
                else 0.0
            )


        elif domain == "AI":

            score = (
                1.00
                if "cs.AI" in categories
                else 0.0
            )

            score += (
                0.70
                if "cs.LG" in categories
                else 0.0
            )

            score += (
                0.40
                if "cs.NE" in categories
                else 0.0
            )

            score += (
                0.35
                if "cs.CL" in categories
                else 0.0
            )


        else:

            score = sum(
                0.20
                for category in categories
                if category in RELEVANT_CATEGORIES
            )


        return min(
            score,
            1.0
        )


    # ========================================================
    # CONTENT SCORE
    # ========================================================

    def _content_score(
        self,
        result: dict,
        query: str
    ) -> float:

        title = normalize(
            result.get(
                "title",
                ""
            )
        )

        text = normalize(
            result_text(result)
        )

        categories = normalize(
            result.get(
                "categories",
                ""
            )
        )


        q_terms = tokens(
            query
        )


        if not q_terms:

            return 0.0


        title_terms = tokens(
            title
        )

        content_terms = tokens(
            f"{categories} {text}"
        )


        title_overlap = (
            len(
                q_terms & title_terms
            )
            /
            len(q_terms)
        )


        content_overlap = (
            len(
                q_terms & content_terms
            )
            /
            len(q_terms)
        )


        phrase_bonus = (
            0.35
            if normalize(query) in title
            else 0.0
        )


        score = (
            title_overlap * 0.55
            +
            content_overlap * 0.30
            +
            phrase_bonus
        )


        return min(
            score,
            1.0
        )


    # ========================================================
    # CONCEPT SCORE
    # ========================================================

    def _concept_score(
        self,
        result: dict,
        query: str
    ) -> float:

        q = normalize(
            query
        )

        title = normalize(
            result.get(
                "title",
                ""
            )
        )

        text = normalize(
            result_text(result)
        )

        combined = (
            f"{title} {text}"
        )


        score = 0.0


        # ----------------------------------------------------
        # Title concept boost
        # ----------------------------------------------------

        query_tokens = tokens(q)
        title_tokens = tokens(title)


        for term in query_tokens:

            if term in title_tokens:

                score += 0.08


        # ----------------------------------------------------
        # Important NLP / AI concept groups
        # ----------------------------------------------------

        concept_groups = [

            {
                "transformer",
                "transformers",
                "self-attention",
                "self attention",
                "multi-head attention",
            },

            {
                "rnn",
                "recurrent neural network",
                "recurrent neural networks",
                "lstm",
                "gru",
            },

            {
                "bert",
                "gpt",
                "llm",
                "large language model",
            },

            {
                "machine learning",
                "deep learning",
                "neural network",
                "neural networks",
            },

        ]


        for group in concept_groups:

            if not any(
                term in q
                for term in group
            ):

                continue


            matches = sum(
                1
                for term in group
                if term in combined
            )


            score += min(
                matches * 0.025,
                0.18
            )


        return min(
            score,
            0.35
        )


    # ========================================================
    # MAIN SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> list[dict]:

        query = str(
            query or ""
        ).strip()


        if not query:

            return []


        # ----------------------------------------------------
        # Detect research domain
        # ----------------------------------------------------

        domain = self.detect_domain(
            query
        )


        # ----------------------------------------------------
        # Create query embedding
        # ----------------------------------------------------

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(
            "float32"
        )


        # ----------------------------------------------------
        # Retrieve wider FAISS candidate pool
        # ----------------------------------------------------

        candidate_k = min(
            CANDIDATE_K,
            self.index.ntotal
        )


        scores, indices = self.index.search(
            embedding,
            candidate_k
        )


        candidates = []


        # ----------------------------------------------------
        # Rerank candidates
        # ----------------------------------------------------

        for semantic_score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:

                continue


            result = dict(
                self.metadata[
                    int(idx)
                ]
            )


            semantic_score = float(
                semantic_score
            )


            category = self._category_score(
                result,
                domain
            )


            content = self._content_score(
                result,
                query
            )


            concept = self._concept_score(
                result,
                query
            )


            # ------------------------------------------------
            # Final reranking score
            # ------------------------------------------------

            final_score = (
                semantic_score * 0.60
                +
                category * 0.20
                +
                content * 0.15
                +
                concept * 0.05
            )


            result.update({

                "similarity":
                    semantic_score,

                "category_score":
                    category,

                "content_score":
                    content,

                "concept_score":
                    concept,

                "final_score":
                    final_score,

                "detected_domain":
                    domain,

                "paper_id":
                    paper_id(result),

                "paper_url":
                    paper_url(result),

            })


            candidates.append(
                result
            )


        # ----------------------------------------------------
        # Restrict NLP / AI results to relevant CS categories
        # ----------------------------------------------------

        if domain in {
            "NLP",
            "AI"
        }:

            filtered = [

                result

                for result in candidates

                if any(
                    category
                    in RELEVANT_CATEGORIES

                    for category
                    in str(
                        result.get(
                            "categories",
                            ""
                        )
                    ).split()
                )

            ]


            if len(filtered) >= top_k:

                candidates = filtered


        # ----------------------------------------------------
        # Sort by final reranking score
        # ----------------------------------------------------

        candidates.sort(
            key=lambda x: x[
                "final_score"
            ],
            reverse=True
        )


        # ----------------------------------------------------
        # FINAL TOP-3
        # ----------------------------------------------------

        return candidates[
            :top_k
        ]


    # ========================================================
    # PAPER SEARCH
    # ========================================================

    def search_papers(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> list[dict]:

        return self.search(
            query,
            top_k=top_k
        )


    # ========================================================
    # FIND PAPER BY ARXIV ID
    # ========================================================

    def find_paper(
        self,
        pid: str
    ):

        target = str(
            pid or ""
        ).strip().lower()


        for result in self.metadata:

            if paper_id(
                result
            ).lower() == target:

                result = dict(
                    result
                )


                result[
                    "paper_id"
                ] = paper_id(
                    result
                )


                result[
                    "paper_url"
                ] = paper_url(
                    result
                )


                return result


        return None