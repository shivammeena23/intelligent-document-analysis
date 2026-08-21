from typing import List, Dict


def build_context(
    documents: List[Dict]
) -> str:
    """
    Convert retrieved documents into a citation-aware
    context string for the LLM.

    Each retrieved document receives a stable source
    number such as [1], [2], [3].
    """

    if not documents:
        return ""

    context_parts = []

    for i, document in enumerate(
        documents,
        start=1
    ):

        source = document.get(
            "source",
            "Unknown"
        )

        page = document.get(
            "page",
            "Unknown"
        )

        chunk_id = document.get(
            "chunk_id",
            f"chunk_{i}"
        )

        text = document.get(
            "text",
            ""
        ).strip()

        context_parts.append(
            f"""
[SOURCE {i}]
Source ID: [{i}]
Document: {source}
Page: {page}
Chunk ID: {chunk_id}

{text}
"""
        )

    return "\n-------------------------\n".join(
        context_parts
    )