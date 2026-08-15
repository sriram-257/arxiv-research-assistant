# ArXiv Research Assistant

A research-focused conversational RAG chatbot for exploring scientific
papers from the ArXiv dataset. The application uses semantic retrieval,
FAISS vector search, a local open-source LLM through Ollama, and a
Streamlit interface.

## Objective

The system is designed to: - Answer complex Computer Science, AI and NLP
research questions. - Retrieve relevant ArXiv papers. - Explain
technical concepts using retrieved research evidence. - Support
context-aware follow-up questions. - Provide research paper links and
source references. - Visualize the relationship between questions and
retrieved concepts. - Generate explanations using a local open-source
LLM.

## Dataset

A Computer Science-focused subset of the ArXiv dataset was prepared for
this project.

Current knowledge base: - 50,000 indexed records - 384-dimensional
embeddings - Top-3 semantic retrieval - Computer Science / AI / NLP
categories - Paper metadata and research text

## Architecture

``` text
User Question
     |
     v
Streamlit Interface
     |
     v
BGE-small Embeddings
     |
     v
FAISS Semantic Search
     |
     v
Top-3 Retrieved Papers
     |
     v
Relevance Ranking
     |
     v
Research Evidence Context
     |
     v
Llama 3.2 3B via Ollama
     |
     +----> Grounded Answer
     +----> Paper Sources
     +----> Concept Visualization
```

## Retrieval Pipeline

1.  Query normalization
2.  Domain detection
3.  BGE-small-en-v1.5 embedding
4.  FAISS vector search
5.  Candidate retrieval
6.  Category relevance scoring
7.  Content/title relevance scoring
8.  Concept relevance scoring
9.  Final ranking
10. Top-3 paper selection

Relevant categories include `cs.CL`, `cs.AI`, `cs.LG`, and `cs.NE`.

## Language Model

The application uses Llama 3.2 3B through Ollama for local explanation
generation.

The generation pipeline is designed to: - Use retrieved research
context. - Avoid fabricated paper information. - Cite retrieved evidence
using `[1]`, `[2]`, and `[3]`. - State when retrieved evidence is
insufficient. - Handle comparison questions. - Support follow-up
questions.

## Conversational RAG

Example:

``` text
User: What is a Transformer model?

Assistant: Explains the Transformer using retrieved research evidence.

User: How is it different from RNNs?

Assistant: Uses conversational context and retrieves relevant evidence
to provide a comparison.

User: Why is self-attention useful?

Assistant: Continues the research discussion using retrieved evidence.
```

## Features

-   Semantic ArXiv paper retrieval
-   Top-3 relevant paper display
-   Research-grounded answers
-   Local Llama 3.2 3B inference
-   Ollama integration
-   Context-aware follow-up questions
-   Research paper links
-   Source citations
-   Concept visualization
-   Retrieval and generation timing
-   Streamlit interface
-   Loading/generation feedback
-   Computer Science-focused retrieval

## Project Structure

``` text
ArxivResearchChatbot/
|
├── app.py
├── rag_engine.py
├── generate_answer.py
├── requirements.txt
├── .env
├── .gitignore
|
├── filter_arxiv.py
├── embed_arxiv.py
├── build_index.py
├── chunk_arxiv.py
|
├── search_test.py
├── test_embeddings.py
|
└── data/
    ├── arxiv_faiss.index
    ├── arxiv_metadata.jsonl
    ├── arxiv_chunks.jsonl
    ├── arxiv_cs_ai_ml.jsonl
    └── arxiv_embeddings...
```

## Technologies

  Component         Technology
  ----------------- --------------------------------
  Interface         Streamlit
  Dataset           ArXiv
  Embeddings        BAAI/bge-small-en-v1.5
  Embedding size    384
  Vector search     FAISS
  LLM               Llama 3.2 3B
  Local inference   Ollama
  Language          Python
  Retrieval         Retrieval-Augmented Generation
  NLP               Sentence Transformers

## Installation

Create and activate a virtual environment:

``` bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Make sure Ollama is installed and running, then:

``` bash
ollama pull llama3.2:3b
```

Verify:

``` bash
ollama list
```

## Run

From the project directory:

``` bash
streamlit run app.py
```

## Example Questions

``` text
What is a Transformer model?
```

``` text
How is a Transformer different from RNNs?
```

``` text
Why is self-attention useful?
```

``` text
Compare Transformers and RNNs.
```

``` text
Explain attention mechanisms in neural networks.
```

## Current Configuration

``` text
Indexed Records       : 50,000
Embedding Dimension   : 384
Embedding Model       : BAAI/bge-small-en-v1.5
Vector Search         : FAISS
Retrieval             : Top-3
Language Model        : Llama 3.2 3B
Inference             : Local Ollama
Interface             : Streamlit
Architecture          : RAG
Domain                : Computer Science / AI / NLP
```

## Grounding and Citations

The answer generator receives the retrieved research evidence as
context. Its instructions require it to: - Answer from the supplied
evidence. - Avoid invented paper titles, IDs, authors, metrics and
results. - Cite evidence with `[1]`, `[2]`, `[3]`. - Prefer the most
relevant sources. - State when evidence is insufficient.

Retrieved papers are displayed separately so users can inspect the
original sources.

## Limitations

-   The current knowledge base is a prepared 50,000-record subset.
-   Llama 3.2 3B is a relatively small local model.
-   Answer quality depends on retrieval quality and the indexed
    evidence.
-   The chatbot should be used as a research assistant, not as a
    replacement for reading original papers.
-   Some questions may not have sufficient evidence in the indexed
    subset.

## Assignment Requirement Mapping

  -----------------------------------------------------------------------
  Requirement                         Implementation
  ----------------------------------- -----------------------------------
  Domain-specific chatbot             Computer Science / AI / NLP ArXiv
                                      assistant

  ArXiv dataset                       Computer Science-focused ArXiv
                                      subset

  Scientific paper retrieval          FAISS semantic search

  Advanced NLP                        Sentence Transformer embeddings and
                                      relevance scoring

  Information extraction              Paper metadata and research-text
                                      retrieval

  Explanation generation              Llama 3.2 3B

  Open-source LLM                     Llama 3.2 3B through Ollama

  Complex queries                     Research-focused RAG prompting

  Follow-up questions                 Conversational RAG

  Paper searching                     Streamlit search interface

  Paper links                         Direct ArXiv links

  Concept visualization               Concept visualization section

  Streamlit                           Complete Streamlit application
  -----------------------------------------------------------------------

## Expected Outcome

The completed system demonstrates a domain-specific scientific research
assistant capable of retrieving research papers, answering advanced
questions, explaining concepts, comparing approaches, supporting
follow-up questions, providing source links, and using an open-source
local LLM within a complete RAG pipeline.

## Project Stack

**ArXiv + BGE Embeddings + FAISS + RAG + Llama 3.2 3B + Ollama +
Streamlit**
