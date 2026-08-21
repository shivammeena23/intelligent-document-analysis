from typing import List, Dict

import faiss
import numpy as np


class FAISSRetriever:
    """
    FAISS-based semantic similarity retriever.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

        # Inner product works as cosine similarity
        # because our embeddings are normalized.
        self.index = faiss.IndexFlatIP(dimension)

        # Store original chunks separately.
        self.documents: List[Dict] = []

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict]
    ):
        """
        Add document embeddings and their metadata to FAISS.
        """

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents."
            )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for the most semantically similar documents.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):
            if index == -1:
                continue

            result = self.documents[index].copy()

            result["score"] = float(score)

            results.append(result)

        return results