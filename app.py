import html
import json
import re
import time
from typing import Any

import requests
import streamlit as st

from rag_engine import ArxivRAG


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

OLLAMA_MODEL = "llama3.2:3b"

TOP_K = 3

MAX_CHARS_PER_SOURCE = 2200
MAX_OUTPUT_TOKENS = 420
TEMPERATURE = 0.10


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0b0e14;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #9ca8ba;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .ready-box {
        background: #103b2c;
        border: 1px solid #1c7658;
        color: #6ee7b7;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 15px 0 20px 0;
        font-weight: 600;
    }

    .metric-box {
        background: #11151d;
        border: 1px solid #303746;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        min-height: 105px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
    }

    .metric-label {
        color: #8995a7;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    .source-card {
        border: 1px solid #303746;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        background: #11151d;
    }

    .source-title {
        font-weight: 700;
        font-size: 1rem;
    }

    .muted {
        color: #8995a7;
        font-size: 0.82rem;
    }

    .concept-box {
        background: #11151d;
        border: 1px solid #303746;
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
    }

    .status-text {
        color: #9ca8ba;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "topic" not in st.session_state:
    st.session_state.topic = ""

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

if "paper_search_results" not in st.session_state:
    st.session_state.paper_search_results = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def get_title(result: dict) -> str:
    return clean_text(
        result.get("title")
        or "Untitled paper"
    )


def get_paper_id(result: dict) -> str:
    value = (
        result.get("paper_id")
        or result.get("arxiv_id")
        or result.get("id")
        or ""
    )

    return clean_text(value).replace(
        "arXiv:",
        ""
    )


def get_text(result: dict) -> str:
    return clean_text(
        result.get("text")
        or result.get("chunk")
        or result.get("abstract")
        or result.get("summary")
        or ""
    )


def get_paper_url(result: dict) -> str:

    existing_url = result.get("paper_url")

    if existing_url:
        return str(existing_url)

    paper_id = get_paper_id(result)

    if paper_id:
        return f"https://arxiv.org/abs/{paper_id}"

    return ""


# ============================================================
# FOLLOW-UP QUESTION DETECTION
# ============================================================

def is_follow_up(question: str) -> bool:

    if not st.session_state.messages:
        return False

    q = question.lower().strip()

    patterns = (
        "how is",
        "how does",
        "how do",
        "why is",
        "why does",
        "why do",
        "what about",
        "and ",
        "also ",
        "does it",
        "is it",
        "can it",
        "what is its",
        "what are its",
        "tell me more",
        "explain further",
        "explain more",
        "compare",
        "difference",
        "different",
        "advantages",
        "disadvantages",
        "limitations",
    )

    return q.startswith(patterns)


# ============================================================
# SEARCH QUERY RESOLUTION
# ============================================================

def prepare_search_query(question: str):

    previous_topic = st.session_state.topic

    follow_up = is_follow_up(question)

    if follow_up and previous_topic:

        search_query = (
            f"{previous_topic}. "
            f"Follow-up question: {question}"
        )

        return search_query, True

    return question, False


# ============================================================
# TOPIC EXTRACTION
# ============================================================

def extract_topic(question: str) -> str:

    patterns = [
        r"(?:what is|what are|explain|define|describe)"
        r"\s+(?:a|an|the)?\s*(.+?)(?:\?|$)",

        r"(?:tell me about)\s+(.+?)(?:\?|$)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )

        if match:

            topic = match.group(1).strip(
                " .?"
            )

            if topic:
                return topic

    return question.rstrip(
        "?."
    )[:120]


# ============================================================
# OLLAMA CONNECTION CHECK
# ============================================================

def check_ollama():

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=3,
        )

        if not response.ok:

            return (
                False,
                "Ollama is not responding."
            )

        data = response.json()

        models = {
            str(model.get("name", ""))
            for model in data.get(
                "models",
                []
            )
        }

        if OLLAMA_MODEL in models:

            return (
                True,
                "Ollama connected"
            )

        base_model = OLLAMA_MODEL.split(":")[0]

        if any(
            model.startswith(base_model + ":")
            for model in models
        ):

            return (
                True,
                "Ollama connected"
            )

        return (
            False,
            f"{OLLAMA_MODEL} is not installed. "
            f"Run: ollama pull {OLLAMA_MODEL}"
        )

    except requests.RequestException:

        return (
            False,
            "Ollama is not running. "
            "Please start Ollama."
        )

    except Exception as exc:

        return (
            False,
            f"Ollama check failed: {exc}"
        )


# ============================================================
# CONVERSATION CONTEXT
# ============================================================

def build_conversation_context(limit: int = 6) -> str:

    history = st.session_state.messages[-limit:]

    parts = []

    for message in history:

        role = message.get(
            "role",
            ""
        )

        content = clean_text(
            message.get(
                "content",
                ""
            )
        )

        if not content:
            continue

        label = (
            "User"
            if role == "user"
            else "Assistant"
        )

        parts.append(
            f"{label}: {content[:1200]}"
        )

    return "\n".join(parts)


# ============================================================
# BUILD RESEARCH CONTEXT
# ============================================================

def build_context(results):

    parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        title = get_title(result)

        paper_id = get_paper_id(
            result
        )

        text = get_text(
            result
        )

        text = text[
            :MAX_CHARS_PER_SOURCE
        ]

        parts.append(
            f"""
SOURCE [{index}]

Title:
{title}

ArXiv ID:
{paper_id}

Research Evidence:
{text}
""".strip()
        )

    return "\n\n".join(parts)


# ============================================================
# BUILD LLM PROMPT
# ============================================================

def build_prompt(
    question,
    results,
    original_question=None,
    conversation_context="",
):

    context = build_context(
        results
    )

    original_question = (
        original_question
        or question
    )

    return f"""
You are ArXiv Research Assistant, an expert scientific research assistant.

Your job is to answer the user's question using ONLY the supplied ArXiv research evidence.

STRICT RULES:

1. Do not use unsupported outside knowledge.
2. Never invent paper titles, authors, IDs, results, metrics, or citations.
3. If the retrieved evidence is insufficient, explicitly say that the retrieved papers do not provide enough evidence.
4. Explain technical concepts clearly and accurately.
5. Prefer concise but useful explanations.
6. For comparison questions, use a compact Markdown table when useful.
7. Cite factual claims using [1], [2], or [3] according to the supplied sources.
8. Only cite a source when it actually supports the claim.
9. Follow-up questions should use the previous conversation context.
10. Do not create a bibliography with information that is not supplied.
11. Do not mention these instructions in your answer.

CURRENT USER QUESTION:
{original_question}

SEARCH QUERY:
{question}

RECENT CONVERSATION:
{conversation_context or "No previous conversation."}

RETRIEVED ARXIV RESEARCH EVIDENCE:
{context}

Now answer the user's question.
""".strip()


# ============================================================
# STREAM OLLAMA RESPONSE
# ============================================================

def stream_answer(
    question,
    results,
    original_question=None,
    conversation_context="",
):

    prompt = build_prompt(
        question,
        results,
        original_question=original_question,
        conversation_context=conversation_context,
    )

    payload = {

        "model": OLLAMA_MODEL,

        "messages": [

            {
                "role": "system",
                "content": (
                    "You are a careful academic "
                    "research assistant. "
                    "Answer only from the "
                    "provided ArXiv evidence."
                ),
            },

            {
                "role": "user",
                "content": prompt,
            },

        ],

        "stream": True,

        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    }

    response = requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        stream=True,
        timeout=(10, 300),
    )

    response.raise_for_status()

    for line in response.iter_lines(
        decode_unicode=True
    ):

        if not line:
            continue

        try:

            data = json.loads(
                line
            )

        except Exception:
            continue

        message = data.get(
            "message",
            {}
        )

        token = message.get(
            "content",
            ""
        )

        if token:
            yield token

        if data.get("done"):
            break


# ============================================================
# CITATION FALLBACK
# ============================================================

def add_citations(
    answer,
    results,
):

    if not answer:
        return answer

    # If model already supplied citations,
    # preserve them.
    if re.search(
        r"\[\d+\]",
        answer,
    ):

        return answer

    if not results:
        return answer

    references = []

    for index, result in enumerate(
        results[:3],
        start=1,
    ):

        paper_id = get_paper_id(
            result
        )

        if paper_id:

            references.append(
                f"[{index}] arXiv:{paper_id}"
            )

    if not references:
        return answer

    return (
        answer
        + "\n\n**Retrieved sources:** "
        + "  ".join(references)
    )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    if not results:
        return

    st.markdown(
        "### 📚 Retrieved Research Papers"
    )

    for index, result in enumerate(
        results[:TOP_K],
        start=1,
    ):

        title = get_title(
            result
        )

        paper_id = get_paper_id(
            result
        )

        similarity = float(
            result.get(
                "similarity",
                0,
            )
        )

        final_score = float(
            result.get(
                "final_score",
                0,
            )
        )

        url = get_paper_url(
            result
        )

        st.markdown(
            f"""
            <div class="source-card">

            <div class="source-title">
            [{index}] {html.escape(title)}
            </div>

            <div class="muted">
            arXiv:{html.escape(paper_id)}
            &nbsp; • &nbsp;
            Semantic similarity: {similarity:.3f}
            &nbsp; • &nbsp;
            Final relevance: {final_score:.3f}
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if url:

            st.link_button(
                f"🔗 Open ArXiv Paper {index}",
                url,
                key=f"open_paper_{index}_{paper_id}",
            )


# ============================================================
# CONCEPT VISUALIZATION
# ============================================================

def concept_visualization(
    question,
    results,
):

    q = question.lower()

    concept_map = [

        (
            "transformer",
            "Transformer",
        ),

        (
            "attention",
            "Self-Attention",
        ),

        (
            "rnn",
            "RNN",
        ),

        (
            "lstm",
            "LSTM",
        ),

        (
            "gru",
            "GRU",
        ),

        (
            "bert",
            "BERT",
        ),

        (
            "gpt",
            "GPT",
        ),

        (
            "llm",
            "Large Language Model",
        ),

        (
            "embedding",
            "Embeddings",
        ),

        (
            "machine translation",
            "Machine Translation",
        ),

        (
            "classification",
            "Classification",
        ),

        (
            "reinforcement learning",
            "Reinforcement Learning",
        ),

        (
            "neural network",
            "Neural Networks",
        ),

    ]

    concepts = []

    for key, label in concept_map:

        if key in q:

            concepts.append(
                label
            )

    concepts = list(
        dict.fromkeys(
            concepts
        )
    )[:6]

    if not concepts:
        return

    st.markdown(
        "### 🧠 Concept Visualization"
    )

    concept_text = "\n".join(
        f"• {concept}"
        for concept in concepts
    )

    st.markdown(
        f"""
        <div class="concept-box">

        <b>Research Question</b>

        ↓

        <br>

        {html.escape(concept_text).replace(chr(10), "<br>")}

        <br><br>

        ↓

        <br>

        <b>Retrieved ArXiv Evidence ({len(results)} papers)</b>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD RAG ENGINE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_rag():

    return ArxivRAG()


try:

    rag = load_rag()

    try:

        stats = rag.stats()

    except Exception:

        stats = {
            "records": getattr(
                rag.index,
                "ntotal",
                0,
            ),

            "dimensions": getattr(
                rag.index,
                "d",
                0,
            ),
        }

    rag_error = None

except Exception as exc:

    rag = None

    stats = {}

    rag_error = str(exc)


# ============================================================
# CHECK OLLAMA
# ============================================================

ollama_ok, ollama_message = check_ollama()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔬 ArXiv Assistant"
    )

    st.caption(
        "Research-focused conversational "
        "RAG chatbot for scientific papers."
    )

    st.divider()

    st.markdown(
        "### 📚 Knowledge Base"
    )

    st.metric(
        "Indexed Records",
        f"{stats.get('records', 0):,}",
    )

    st.metric(
        "Embedding Dimension",
        stats.get(
            "dimensions",
            "—",
        ),
    )

    st.metric(
        "Retrieval Top-K",
        TOP_K,
    )

    st.divider()

    st.markdown(
        "### 🤖 Local LLM"
    )

    st.write(
        OLLAMA_MODEL
    )

    if ollama_ok:

        st.success(
            "Ollama ready"
        )

    else:

        st.error(
            ollama_message
        )

    st.divider()

    st.markdown(
        "### ⚙️ Architecture"
    )

    st.write(
        "• ArXiv Computer Science subset"
    )

    st.write(
        "• BGE-small embeddings"
    )

    st.write(
        "• FAISS semantic retrieval"
    )

    st.write(
        "• Top-3 research retrieval"
    )

    st.write(
        "• Conversational RAG"
    )

    st.write(
        "• Local Llama 3.2 3B"
    )

    st.write(
        "• Research citations"
    )

    st.write(
        "• Paper search"
    )

    st.write(
        "• Concept visualization"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.topic = ""

        st.session_state.last_latency = None

        st.session_state.paper_search_results = []

        st.rerun()


# ============================================================
# ERROR CHECK
# ============================================================

if rag_error:

    st.error(
        "RAG engine could not start:\n\n"
        + rag_error
    )

    st.stop()


if not ollama_ok:

    st.warning(
        ollama_message
    )

    st.stop()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">🔬 ArXiv Research Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">'
    'Explore scientific research using semantic retrieval, '
    'conversational RAG, and a local open-source language model.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown(
    f"""
    <div class="ready-box">

    ✓ System ready
    &nbsp; • &nbsp;
    FAISS: {stats.get("records", 0):,} records
    &nbsp; • &nbsp;
    Embedding: {stats.get("dimensions", "—")}D
    &nbsp; • &nbsp;
    Top-K: {TOP_K}
    &nbsp; • &nbsp;
    LLM: {OLLAMA_MODEL}

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.markdown(
        f"""
        <div class="metric-box">

        <div class="metric-value">
        {stats.get("records", 0):,}
        </div>

        <div class="metric-label">
        Indexed Records
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric2:

    st.markdown(
        f"""
        <div class="metric-box">

        <div class="metric-value">
        {stats.get("dimensions", "—")}
        </div>

        <div class="metric-label">
        Embedding Dimensions
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric3:

    st.markdown(
        f"""
        <div class="metric-box">

        <div class="metric-value">
        {TOP_K}
        </div>

        <div class="metric-label">
        Retrieved Sources
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric4:

    st.markdown(
        """
        <div class="metric-box">

        <div class="metric-value">
        Local
        </div>

        <div class="metric-label">
        LLM Processing
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ============================================================
# PAPER SEARCH
# ============================================================

with st.expander(
    "🔎 Search ArXiv Papers",
    expanded=False,
):

    paper_query = st.text_input(
        "Paper search",
        placeholder=(
            "Example: transformer machine translation"
        ),
        key="paper_search_input",
    )

    search_button = st.button(
        "Search Papers",
        type="primary",
    )

    if search_button:

        if not paper_query.strip():

            st.warning(
                "Enter a paper search query."
            )

        else:

            with st.spinner(
                "Searching the ArXiv knowledge base..."
            ):

                search_results = rag.search_papers(
                    paper_query,
                    top_k=TOP_K,
                )

            st.session_state.paper_search_results = (
                search_results
            )


    if st.session_state.paper_search_results:

        st.markdown(
            "### Search Results"
        )

        for index, result in enumerate(
            st.session_state.paper_search_results,
            start=1,
        ):

            title = get_title(
                result
            )

            paper_id = get_paper_id(
                result
            )

            similarity = float(
                result.get(
                    "similarity",
                    0,
                )
            )

            url = get_paper_url(
                result
            )

            st.markdown(
                f"""
                **[{index}] {html.escape(title)}**

                `arXiv:{html.escape(paper_id)}`

                Semantic similarity: `{similarity:.3f}`
                """,
                unsafe_allow_html=True,
            )

            if url:

                st.link_button(
                    "Open paper",
                    url,
                    key=f"search_open_{index}_{paper_id}",
                )

            st.divider()


# ============================================================
# DISPLAY EXISTING CONVERSATION
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role"
    )

    if role == "user":

        with st.chat_message(
            "user",
            avatar="👤",
        ):

            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )

    elif role == "assistant":

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )

            if message.get(
                "results"
            ):

                display_sources(
                    message["results"]
                )

                concept_visualization(
                    message.get(
                        "search_query",
                        "",
                    ),
                    message["results"],
                )

            if message.get(
                "latency"
            ) is not None:

                st.caption(
                    f"Retrieval: "
                    f"{message.get('retrieval_latency', 0):.2f}s"
                    " • "
                    f"Generation: "
                    f"{message.get('generation_latency', 0):.1f}s"
                    " • "
                    f"Total: "
                    f"{message.get('latency', 0):.1f}s"
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a research question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = clean_text(
        question
    )

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # --------------------------------------------------------
    # SHOW USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message(
        "user",
        avatar="👤",
    ):

        st.markdown(
            question
        )


    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # --------------------------------------------------------
    # CONVERSATION CONTEXT
    # --------------------------------------------------------

    conversation_context = (
        build_conversation_context(
            limit=6
        )
    )


    # --------------------------------------------------------
    # FOLLOW-UP / SEARCH QUERY
    # --------------------------------------------------------

    search_query, follow_up = (
        prepare_search_query(
            question
        )
    )


    if not follow_up:

        st.session_state.topic = (
            extract_topic(
                question
            )
        )


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieval_start = (
        time.perf_counter()
    )

    with st.status(
        "🔎 Searching ArXiv knowledge base...",
        expanded=True,
    ) as retrieval_status:

        st.write(
            "🔍 Creating semantic query..."
        )

        results = rag.search(
            search_query,
            top_k=TOP_K,
        )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        retrieval_status.update(
            label=(
                f"✓ Retrieved "
                f"{len(results)} relevant papers"
            ),
            state="complete",
            expanded=False,
        )


    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        answer = (
            "I could not find relevant "
            "research papers in the ArXiv "
            "knowledge base for this question."
        )

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            st.markdown(
                answer
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "results": [],
                "search_query": search_query,
                "latency": retrieval_latency,
                "retrieval_latency": retrieval_latency,
                "generation_latency": 0.0,
            }
        )

        st.rerun()


    # --------------------------------------------------------
    # GENERATION
    # IMPORTANT:
    # EVERYTHING IS INSIDE THE ASSISTANT CHAT CONTAINER
    # --------------------------------------------------------

    generation_start = (
        time.perf_counter()
    )

    collected_tokens = []

    with st.chat_message(
        "assistant",
        avatar="🤖",
    ):

        st.markdown(
            "### 🧠 Research Answer"
        )

        answer_placeholder = st.empty()

        status_placeholder = st.empty()

        status_placeholder.info(
            "📚 Research context loaded • "
            "🤖 Llama 3.2 3B is generating..."
        )

        try:

            for token in stream_answer(
                search_query,
                results,
                original_question=question,
                conversation_context=conversation_context,
            ):

                collected_tokens.append(
                    token
                )

                current_answer = (
                    "".join(
                        collected_tokens
                    )
                )

                answer_placeholder.markdown(
                    current_answer + "▌"
                )


            answer = (
                "".join(
                    collected_tokens
                )
                .strip()
            )


            if not answer:

                answer = (
                    "The local Llama model "
                    "returned an empty response. "
                    "Please try again."
                )


            answer = add_citations(
                answer,
                results,
            )


            answer_placeholder.markdown(
                answer
            )


            status_placeholder.success(
                "✓ Research answer generated"
            )


        except requests.exceptions.Timeout:

            answer = (
                "⏱️ The local Llama model "
                "took too long to respond. "
                "Please try again."
            )

            answer_placeholder.error(
                answer
            )


        except requests.exceptions.ConnectionError:

            answer = (
                "❌ Could not connect to Ollama. "
                "Please make sure Ollama is running."
            )

            answer_placeholder.error(
                answer
            )


        except requests.exceptions.HTTPError as exc:

            answer = (
                "❌ Ollama returned an HTTP error:\n\n"
                f"{exc}"
            )

            answer_placeholder.error(
                answer
            )


        except Exception as exc:

            answer = (
                "❌ An error occurred while "
                "generating the answer:\n\n"
                f"{exc}"
            )

            answer_placeholder.error(
                answer
            )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        display_sources(
            results
        )


        # ----------------------------------------------------
        # CONCEPT VISUALIZATION
        # ----------------------------------------------------

        concept_visualization(
            question,
            results,
        )


    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    generation_latency = (
        time.perf_counter()
        - generation_start
    )

    total_latency = (
        retrieval_latency
        + generation_latency
    )


    st.caption(
        f"Retrieval: "
        f"{retrieval_latency:.2f}s"
        " • "
        f"Generation: "
        f"{generation_latency:.1f}s"
        " • "
        f"Total: "
        f"{total_latency:.1f}s"
    )


    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "results": results,
            "search_query": search_query,
            "latency": total_latency,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
        }
    )

    st.session_state.last_latency = (
        total_latency
    )


    # --------------------------------------------------------
    # RERUN TO BUILD STABLE CHAT HISTORY
    # --------------------------------------------------------

    st.rerun()