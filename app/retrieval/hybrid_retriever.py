from typing import List, Dict


class HybridRetriever:
    """
    Combines semantic FAISS retrieval with BM25 keyword retrieval.
    """

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever
    ):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever

    def search(
        self,
        query: str,
        query_embedding,
        top_k: int = 5,
        semantic_k: int = 10,
        keyword_k: int = 10
    ) -> List[Dict]:

        semantic_results = self.faiss_retriever.search(
            query_embedding,
            top_k=semantic_k
        )

        keyword_results = self.bm25_retriever.search(
            query,
            top_k=keyword_k
        )

        combined = {}

        # Add semantic results.
        for rank, result in enumerate(semantic_results):

            chunk_id = result["chunk_id"]

            combined.setdefault(
                chunk_id,
                {
                    "document": result,
                    "semantic_rank": None,
                    "keyword_rank": None
                }
            )

            combined[chunk_id]["semantic_rank"] = rank + 1

        # Add keyword results.
        for rank, result in enumerate(keyword_results):

            chunk_id = result["chunk_id"]

            combined.setdefault(
                chunk_id,
                {
                    "document": result,
                    "semantic_rank": None,
                    "keyword_rank": None
                }
            )

            combined[chunk_id]["keyword_rank"] = rank + 1

        # Reciprocal Rank Fusion.
        #
        # RRF score:
        # 1 / (60 + rank)
        #
        # 60 is a standard smoothing constant.

        for item in combined.values():

            score = 0.0

            if item["semantic_rank"] is not None:
                score += 1 / (
                    60 + item["semantic_rank"]
                )

            if item["keyword_rank"] is not None:
                score += 1 / (
                    60 + item["keyword_rank"]
                )

            item["fusion_score"] = score

        ranked = sorted(
            combined.values(),
            key=lambda x: x["fusion_score"],
            reverse=True
        )

        results = []

        for item in ranked[:top_k]:

            document = item["document"].copy()

            document["fusion_score"] = item["fusion_score"]
            document["semantic_rank"] = item["semantic_rank"]
            document["keyword_rank"] = item["keyword_rank"]

            results.append(document)

        return results