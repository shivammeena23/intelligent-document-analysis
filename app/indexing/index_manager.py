import json
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np


class IndexManager:
    """
    Handles creation, persistence and loading of
    the FAISS vector index and document chunks.
    """

    def __init__(
        self,
        index_directory: str = "data/index"
    ):
        self.index_directory = Path(index_directory)

        self.index_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.faiss_path = (
            self.index_directory / "faiss.index"
        )

        self.chunks_path = (
            self.index_directory / "chunks.pkl"
        )

        self.metadata_path = (
            self.index_directory / "metadata.json"
        )

    # -----------------------------------------------------
    # Save index
    # -----------------------------------------------------

    def save(
        self,
        index: faiss.Index,
        chunks: List[Dict]
    ):
        """
        Save FAISS index and document chunks to disk.
        """

        faiss.write_index(
            index,
            str(self.faiss_path)
        )

        with open(
            self.chunks_path,
            "wb"
        ) as file:

            pickle.dump(
                chunks,
                file
            )

        metadata = {
            "num_vectors": index.ntotal,
            "dimension": index.d,
            "num_chunks": len(chunks)
        }

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        print(
            "\nIndex saved successfully."
        )

        print(
            f"FAISS index : {self.faiss_path}"
        )

        print(
            f"Chunks      : {self.chunks_path}"
        )

        print(
            f"Metadata    : {self.metadata_path}"
        )

    # -----------------------------------------------------
    # Load index
    # -----------------------------------------------------

    def load(self):
        """
        Load FAISS index and chunks from disk.
        """

        if not self.exists():
            raise FileNotFoundError(
                "Saved index does not exist. "
                "Run the indexing pipeline first."
            )

        index = faiss.read_index(
            str(self.faiss_path)
        )

        with open(
            self.chunks_path,
            "rb"
        ) as file:

            chunks = pickle.load(
                file
            )

        print(
            "\nIndex loaded successfully."
        )

        print(
            f"Vectors : {index.ntotal}"
        )

        print(
            f"Chunks  : {len(chunks)}"
        )

        print(
            f"Dimension : {index.d}"
        )

        return index, chunks

    # -----------------------------------------------------
    # Check whether index exists
    # -----------------------------------------------------

    def exists(self) -> bool:
        """
        Check whether all required index files exist.
        """

        return (
            self.faiss_path.exists()
            and
            self.chunks_path.exists()
            and
            self.metadata_path.exists()
        )