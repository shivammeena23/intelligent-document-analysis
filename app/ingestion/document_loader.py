from pathlib import Path
from typing import List, Dict

from app.ingestion.pdf_loader import load_pdf


def load_documents(
    directory: str = "data/documents"
) -> List[Dict]:
    """
    Load all PDF documents from a directory.
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {directory}"
        )

    pdf_files = sorted(
        directory_path.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {directory}"
        )

    all_documents = []

    print(
        f"\nFound {len(pdf_files)} PDF document(s)."
    )

    for pdf_file in pdf_files:

        print(
            f"\nLoading: {pdf_file.name}"
        )

        documents = load_pdf(
            str(pdf_file)
        )

        all_documents.extend(
            documents
        )

        print(
            f"Pages extracted: {len(documents)}"
        )

    print(
        f"\nTotal pages extracted: "
        f"{len(all_documents)}"
    )

    return all_documents