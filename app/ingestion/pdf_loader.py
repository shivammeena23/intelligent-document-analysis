from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF


def load_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF page by page.

    Returns:
        List of dictionaries containing:
        - text
        - page
        - source
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided file is not a PDF.")

    documents = []

    pdf = fitz.open(file_path)

    for page_number, page in enumerate(pdf):
        text = page.get_text("text").strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "page": page_number + 1,
                "source": path.name,
            }
        )

    pdf.close()

    return documents


if __name__ == "__main__":
    pdf_path = "data/documents/test.pdf"

    pages = load_pdf(pdf_path)

    print(f"Extracted {len(pages)} pages.\n")

    for page in pages[:3]:
        print("=" * 80)
        print(f"Source: {page['source']}")
        print(f"Page: {page['page']}")
        print("=" * 80)
        print(page["text"][:1000])
        print()