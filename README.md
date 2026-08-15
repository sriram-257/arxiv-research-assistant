# 🔬 ArXiv Research Assistant

A domain-specific conversational **Retrieval-Augmented Generation (RAG)** chatbot for exploring scientific research from the ArXiv dataset.

The application combines semantic embeddings, FAISS vector retrieval, relevance scoring, a local Llama 3.2 3B model through Ollama, and a Streamlit interface to answer complex research questions, explain concepts, retrieve papers, and handle context-aware follow-up questions.

## 🎯 Assignment Objective

The project implements a chatbot that can:

- Answer complex questions in a selected scientific domain.
- Retrieve relevant research papers from an ArXiv knowledge base.
- Explain technical concepts using retrieved research evidence.
- Provide research-oriented summaries and comparisons.
- Handle follow-up questions using conversation context.
- Provide source references and direct ArXiv links.
- Include concept visualization.
- Use an open-source/local LLM for explanation generation.
- Run through an interactive Streamlit application.

## 📚 Dataset and Knowledge Base

The project uses the Cornell University ArXiv dataset.

A Computer Science-focused subset was prepared for the application, with emphasis on:

- `cs.CL` — Computation and Language
- `cs.AI` — Artificial Intelligence
- `cs.LG` — Machine Learning
- `cs.NE` — Neural and Evolutionary Computing

### Current indexed knowledge base

- **50,000 records**
- **384-dimensional embeddings**
- **Top-3 retrieval**
- Paper metadata and research text
- FAISS vector index

> **Important:** The generated 50,000-record data/index artifacts are intentionally not stored in this Git repository because they are large (the complete local data directory is approximately 726 MB, including a ~467 MB embeddings file). The full generated data remains part of the local/assignment project package. To run the application from a fresh clone, place the required prebuilt files in the `data/` directory as described below.

## 🧠 System Architecture

```text
                    ArXiv Dataset
                         │
                         ▼
             Computer Science Subset
                         │
                         ▼
                Text + Metadata
                         │
                         ▼
             BGE-small Embeddings
                         │
                         ▼
                  FAISS Index
                         │
                         │
User Question ───────────┘
      │
      ▼
Question Embedding
      │
      ▼
FAISS Semantic Search
      │
      ▼
Candidate Retrieval + Relevance Scoring
      │
      ▼
Top 3 Relevant ArXiv Papers
      │
      ▼
Research Evidence Context
      │
      ▼
Llama 3.2 3B via Ollama
      │
      ▼
Research-Grounded Answer
      │
      ├──────────► Source Citations
      ├──────────► ArXiv Paper Links
      └──────────► Concept Visualization
```

## 🔎 Retrieval Pipeline

The retrieval system performs:

1. Query normalization
2. Domain detection
3. BGE-small-en-v1.5 embedding generation
4. FAISS semantic similarity search
5. Candidate retrieval
6. Category relevance scoring
7. Title/content overlap scoring
8. Concept relevance scoring
9. Final relevance ranking
10. Top-3 paper selection

## 🤖 Language Model

The application uses **Llama 3.2 3B** through **Ollama** for local answer generation.

The model is **not fine-tuned on the ArXiv dataset**. The project uses Retrieval-Augmented Generation: relevant ArXiv evidence is retrieved first and then supplied to the LLM for explanation generation.

The generation prompt is designed to:

- Use supplied research context.
- Avoid unsupported claims.
- Avoid inventing paper titles, IDs, authors, results, or metrics.
- Cite retrieved evidence using `[1]`, `[2]`, and `[3]`.
- State when retrieved evidence is insufficient.
- Provide concise technical explanations.
- Use comparison tables when appropriate.

## 💬 Context-Aware Follow-Up Questions

Example:

```text
User:
What is a Transformer model?

Assistant:
Provides a research-grounded explanation.

User:
How is it different from RNNs?

Assistant:
Uses the previous topic and retrieved research
to provide a Transformer vs RNN comparison.

User:
Why is self-attention useful?

Assistant:
Continues the research discussion using retrieved evidence.
```

## 📄 Research Paper Retrieval

The application displays the **Top 3** retrieved research papers, including paper title, ArXiv ID, relevance information, and direct ArXiv links.

## 🧩 Concept Visualization

The interface connects:

```text
Research Question
       ↓
Detected Concepts
       ↓
Retrieved ArXiv Evidence
```

## 🖥️ Streamlit Interface

Features include:

- Research-focused UI
- Knowledge-base statistics
- Embedding information
- Retrieval Top-K display
- Ollama status
- Research question input
- Loading/generation feedback
- Top-3 paper retrieval
- Research-grounded answers
- Source cards
- Direct ArXiv links
- Follow-up questions
- Paper search
- Concept visualization
- Retrieval and generation timing
- Conversation clearing

## 📁 Project Structure

```text
ArxivResearchChatbot/
│
├── app.py
├── rag_engine.py
├── generate_answer.py
├── build_index.py
├── chunk_arxiv.py
├── embed_arxiv.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── filter_arxiv.py
├── search_test.py
├── test_embeddings.py
│
└── data/
    ├── arxiv_chunks.jsonl
    ├── arxiv_cs_ai_ml.jsonl
    ├── arxiv_embeddings.jsonl
    ├── arxiv_faiss.index
    ├── arxiv_metadata.jsonl
    └── filter_arxiv.py
```

The five large generated data/index files are excluded from GitHub through `.gitignore`.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Dataset | ArXiv |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Embedding Dimension | 384 |
| Vector Search | FAISS |
| Language Model | Llama 3.2 3B |
| Local LLM Runtime | Ollama |
| NLP | Sentence Transformers |
| Retrieval Architecture | RAG |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/sriram-257/arxiv-research-assistant.git
cd arxiv-research-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and prepare Ollama

Make sure Ollama is installed and running.

```bash
ollama pull llama3.2:3b
```

Verify:

```bash
ollama list
```

## 📦 Knowledge-Base Files

The prebuilt 50,000-record artifacts are too large for the normal Git repository and therefore are not included in the GitHub clone.

For the fully runnable assignment package, place these files in `data/`:

```text
data/arxiv_faiss.index
data/arxiv_metadata.jsonl
data/arxiv_chunks.jsonl
data/arxiv_cs_ai_ml.jsonl
data/arxiv_embeddings.jsonl
```

The current local assignment package contains the complete generated knowledge base.

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 🧪 Example Questions

```text
What is a Transformer model?
```

```text
How is it different from RNNs?
```

```text
Why is self-attention useful?
```

```text
Compare Transformers and RNNs.
```

## 📊 Current Configuration

```text
Indexed Records       : 50,000
Embedding Dimension   : 384
Embedding Model       : BAAI/bge-small-en-v1.5
Vector Search         : FAISS
Retrieval             : Top-3
Language Model         : Llama 3.2 3B
Inference              : Local Ollama
Interface              : Streamlit
Architecture          : Retrieval-Augmented Generation
Domain                : Computer Science / AI / NLP
```

## 🛡️ Grounding and Citation Strategy

The answer generator receives retrieved research evidence as context and is instructed to avoid fabricated information, cite supported claims with `[1]`, `[2]`, or `[3]`, prefer relevant sources, and state when evidence is insufficient.

## ⚠️ Limitations

- The current knowledge base is a prepared 50,000-record subset.
- Llama 3.2 3B is a relatively small local model.
- Retrieval quality depends on indexed records and embeddings.
- Local generation speed depends on available hardware.
- The chatbot should assist research rather than replace reading the original paper.
- Questions outside the indexed evidence may not have sufficient support.

## 🚀 Future Improvements

- Hybrid keyword + vector retrieval
- Cross-encoder reranking
- Full-paper PDF ingestion
- Larger local LLMs
- Multi-paper comparison
- Author/topic/year filters
- Advanced research graph visualization
- Retrieval and generation evaluation metrics
- Improved long-term conversation memory

## ✅ Assignment Requirement Mapping

| Assignment Requirement | Implementation |
|---|---|
| Domain-specific chatbot | Computer Science / AI / NLP ArXiv assistant |
| ArXiv dataset | Computer Science-focused ArXiv subset |
| Scientific paper retrieval | FAISS semantic retrieval |
| Advanced NLP | Sentence Transformer embeddings + relevance scoring |
| Information extraction | Paper metadata and research-text retrieval |
| Summarization / explanation | Llama 3.2 3B |
| Open-source LLM | Llama 3.2 3B through Ollama |
| Complex queries | Research-focused RAG prompting |
| Follow-up questions | Context-aware conversational RAG |
| Paper searching | Streamlit paper-search interface |
| Paper links | Direct ArXiv links |
| Concept visualization | Concept visualization section |
| Streamlit | Complete interactive Streamlit application |

## 🎓 Expected Outcome

The system demonstrates a domain-specific scientific research assistant capable of retrieving relevant research papers, answering advanced questions, explaining technical concepts, comparing approaches, supporting follow-up questions, providing source links, and using an open-source local LLM within a complete RAG pipeline.

## 🔗 Repository

https://github.com/sriram-257/arxiv-research-assistant

## 👨‍💻 Project Stack

**ArXiv + BGE Embeddings + FAISS + RAG + Llama 3.2 3B + Ollama + Streamlit**
