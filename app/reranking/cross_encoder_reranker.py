from typing import List, Dict, Optional

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a Cross-Encoder model.

    The reranker produces a relevance score for every
    query-document pair.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        self.model_name = model_name

        print(
            f"Loading reranker model: {model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

        print(
            "Reranker model loaded successfully."
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Dict]:
        """
        Score each query-document pair and return
        the most relevant documents.

        Parameters
        ----------
        query:
            User's question.

        documents:
            Candidate documents from hybrid retrieval.

        top_k:
            Maximum number of documents to return.

        min_score:
            Optional minimum Cross-Encoder score.
            Documents below this score are removed.

            Keep this as None initially while we
            calibrate the threshold on our dataset.
        """

        if not documents:
            return []

        # -------------------------------------------------
        # Create query-document pairs
        # -------------------------------------------------

        pairs = [
            [
                query,
                document["text"]
            ]
            for document in documents
        ]

        # -------------------------------------------------
        # Generate relevance scores
        # -------------------------------------------------

        scores = self.model.predict(
            pairs
        )

        reranked_documents = []

        for document, score in zip(
            documents,
            scores
        ):

            result = document.copy()

            result["reranker_score"] = float(
                score
            )

            reranked_documents.append(
                result
            )

        # -------------------------------------------------
        # Sort by relevance
        # -------------------------------------------------

        reranked_documents.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        # -------------------------------------------------
        # Apply optional relevance threshold
        # -------------------------------------------------

        if min_score is not None:

            reranked_documents = [
                document
                for document in reranked_documents
                if document["reranker_score"] >= min_score
            ]

        # -------------------------------------------------
        # Return top K
        # -------------------------------------------------

        return reranked_documents[:top_k]