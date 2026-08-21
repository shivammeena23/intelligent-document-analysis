from typing import List, Dict


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:
    """
    Split page-level documents into smaller overlapping chunks.

    Args:
        documents: List of page-level document dictionaries.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunk dictionaries with metadata.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []

    for document in documents:

        text = document["text"]
        source = document["source"]
        page = document["page"]

        start = 0
        chunk_number = 1

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source,
                        "page": page,
                        "chunk_id": f"{source}_{page}_{chunk_number}"
                    }
                )

            start += chunk_size - chunk_overlap
            chunk_number += 1

    return chunks