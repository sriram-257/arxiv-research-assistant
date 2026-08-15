import re
import time
from typing import Iterable

import ollama


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "llama3.2:3b"

MAX_CONTEXT_CHARS = 9000
MAX_OUTPUT_TOKENS = 550

TEMPERATURE = 0.05


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ArXiv Research Assistant, a research-focused Retrieval-Augmented
Generation (RAG) assistant for Computer Science, Artificial Intelligence,
Machine Learning, NLP and related scientific research.

Your answers must be grounded strictly in the retrieved research evidence
provided by the application.

IMPORTANT RULES:

1. Use ONLY the supplied retrieved research context.
   Do not use outside knowledge to introduce new factual claims.

2. If the retrieved evidence does not contain enough information to answer
   the question, explicitly say:
   "The retrieved papers do not provide enough evidence to answer this
   confidently."

3. Never invent:
   - paper titles
   - authors
   - arXiv IDs
   - experiments
   - datasets
   - numerical results
   - performance metrics
   - conclusions
   - citations
   - technical claims

4. Every factual statement derived from a retrieved source must have an
   appropriate citation such as [1], [2], or [3].

5. Only cite a source when that source actually supports the statement.

6. Do NOT cite all retrieved papers automatically.

7. Do not create bibliographic information that is not present in the
   retrieved context.

8. When explaining a concept, distinguish between:
   - what the retrieved papers explicitly support
   - a concise explanation based on that evidence

9. Do not confidently state a technical detail if the retrieved evidence
   does not support it.

10. For comparison questions:
    - identify the comparison criteria
    - compare only information supported by the retrieved evidence
    - use a Markdown table when it genuinely improves clarity

11. For questions asking "why", explain the reasons using evidence from the
    retrieved papers.

12. For questions asking "how", explain the mechanism or process only when
    the retrieved evidence supports it.

13. Prefer accurate, concise and academically useful answers over long
    answers.

14. Use clear Markdown formatting:
    - headings when useful
    - numbered lists for steps
    - bullet points for key findings
    - tables for meaningful comparisons

15. Do not add a generic References section.
    The application displays retrieved paper cards separately.

16. If the evidence is weak, incomplete or contradictory, say so explicitly.

17. Never pretend that information came from a paper when it was not present
    in the supplied evidence.

18. Before producing the final answer, internally check:
    - Is every factual claim supported?
    - Are citations attached to the correct claims?
    - Did I invent anything?
    - Did I answer the actual question?
""".strip()


# ============================================================
# SOURCE FORMATTING
# ============================================================

def _source_label(result: dict, rank: int) -> str:
    """
    Convert one retrieved result into a compact evidence block.
    """

    title = str(
        result.get("title")
        or "Untitled paper"
    ).strip()

    paper_id = str(
        result.get("paper_id")
        or result.get("id")
        or ""
    ).strip()

    text = str(
        result.get("text")
        or result.get("chunk")
        or result.get("abstract")
        or ""
    ).strip()

    # Clean excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Avoid unnecessarily huge source blocks
    if len(text) > 1800:
        text = text[:1800] + "..."

    if paper_id:
        return (
            f"[{rank}] {title}\n"
            f"arXiv ID: {paper_id}\n"
            f"Evidence: {text}"
        )

    return (
        f"[{rank}] {title}\n"
        f"Evidence: {text}"
    )


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(results: list[dict]) -> str:
    """
    Build the evidence context supplied to the LLM.
    """

    if not results:
        return "No research papers were retrieved."

    blocks = [
        _source_label(result, index)
        for index, result in enumerate(results, start=1)
    ]

    context = "\n\n".join(blocks)

    return context[:MAX_CONTEXT_CHARS]


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    question: str,
    results: list[dict]
) -> str:

    context = build_context(results)

    return f"""
RESEARCH QUESTION
-----------------
{question}

RETRIEVED RESEARCH EVIDENCE
----------------------------
{context}

TASK
----
Answer the research question using ONLY the retrieved evidence above.

Requirements:

- Answer the actual question directly.
- Use [1], [2], [3], etc. for claims supported by the corresponding evidence.
- Do not invent information.
- Do not introduce facts that are absent from the evidence.
- If evidence is insufficient, clearly state that.
- Prefer accuracy over completeness.
- Keep the answer academically useful and easy to understand.

Before finalizing, verify that every citation actually supports the claim it
is attached to.

FINAL ANSWER:
""".strip()


# ============================================================
# STREAM ANSWER
# ============================================================

def stream_answer(
    question: str,
    results: list[dict]
) -> Iterable[str]:

    prompt = build_prompt(question, results)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True,
        options={
            "temperature": TEMPERATURE,
            "num_predict": MAX_OUTPUT_TOKENS,
            "num_ctx": 4096,
        },
    )

    for chunk in response:

        # Compatible with different Ollama Python response formats
        if isinstance(chunk, dict):

            message = chunk.get("message", {})

            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = getattr(
                    message,
                    "content",
                    ""
                )

        else:

            message = getattr(
                chunk,
                "message",
                None
            )

            if message is not None:
                content = getattr(
                    message,
                    "content",
                    ""
                )
            else:
                content = ""

        if content:
            yield content


# ============================================================
# NON-STREAMING HELPER
# ============================================================

def generate_answer(
    question: str,
    results: list[dict]
) -> tuple[str, float]:

    start = time.perf_counter()

    parts = list(
        stream_answer(
            question,
            results
        )
    )

    answer = "".join(parts).strip()

    elapsed = time.perf_counter() - start

    return answer, elapsed


# ============================================================
# OLLAMA HEALTH CHECK
# ============================================================

def ollama_ready() -> tuple[bool, str]:

    try:

        response = ollama.list()

        if isinstance(response, dict):
            models = response.get("models", [])
        else:
            models = getattr(
                response,
                "models",
                []
            )

        names = set()

        for model in models:

            if isinstance(model, dict):

                name = str(
                    model.get("name")
                    or model.get("model")
                    or ""
                )

            else:

                name = str(
                    getattr(
                        model,
                        "model",
                        ""
                    )
                    or getattr(
                        model,
                        "name",
                        ""
                    )
                )

            if name:
                names.add(name)

        # Exact model match
        if MODEL_NAME in names:
            return True, MODEL_NAME

        # Accept another tag of the same base model
        base_name = MODEL_NAME.split(":")[0]

        for name in names:

            if name.startswith(base_name + ":"):
                return True, name

        return (
            False,
            f"{MODEL_NAME} not found. Run: ollama pull {MODEL_NAME}"
        )

    except Exception as exc:

        return (
            False,
            f"Ollama unavailable: {exc}"
        )