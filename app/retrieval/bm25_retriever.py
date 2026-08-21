from typing import List, Dict

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 keyword-based document retriever.
    """

    def __init__(self, documents: List[Dict]):
        self.documents = documents

        # Tokenize every document chunk.
        tokenized_documents = [
            document["text"].lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            result = self.documents[index].copy()

            result["bm25_score"] = float(scores[index])

            results.append(result)

        return results