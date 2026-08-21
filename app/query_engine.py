from app.embeddings.embedder import EmbeddingModel

from app.retrieval.faiss_retriever import (
    FAISSRetriever
)

from app.retrieval.bm25_retriever import (
    BM25Retriever
)

from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

from app.reranking.cross_encoder_reranker import (
    CrossEncoderReranker
)

from app.generation.context_builder import (
    build_context
)

from app.generation.llm import (
    LLMGenerator
)

from app.indexing.index_manager import (
    IndexManager
)


class QueryEngine:
    """
    Handles query-time retrieval, reranking,
    evidence validation and LLM generation.
    """

    # -----------------------------------------------------
    # Evidence Gate Configuration
    # -----------------------------------------------------

    EVIDENCE_THRESHOLD = -8.0

    def __init__(self):

        print("\nInitializing Query Engine...")

        # -------------------------------------------------
        # Load saved index
        # -------------------------------------------------

        index_manager = IndexManager()

        index, chunks = index_manager.load()

        self.chunks = chunks

        # -------------------------------------------------
        # Embedding model
        # -------------------------------------------------

        self.embedding_model = EmbeddingModel()

        # -------------------------------------------------
        # FAISS Retriever
        # -------------------------------------------------

        self.faiss_retriever = FAISSRetriever(
            dimension=index.d
        )

        self.faiss_retriever.index = index

        self.faiss_retriever.documents = chunks

        # -------------------------------------------------
        # BM25 Retriever
        # -------------------------------------------------

        self.bm25_retriever = BM25Retriever(
            chunks
        )

        # -------------------------------------------------
        # Hybrid Retriever
        # -------------------------------------------------

        self.hybrid_retriever = HybridRetriever(
            faiss_retriever=self.faiss_retriever,
            bm25_retriever=self.bm25_retriever
        )

        # -------------------------------------------------
        # Cross-Encoder Reranker
        # -------------------------------------------------

        self.reranker = CrossEncoderReranker()

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        self.llm = LLMGenerator()

        print(
            "\nQuery Engine initialized."
        )

    # -----------------------------------------------------
    # Evidence Gate
    # -----------------------------------------------------

    def evaluate_evidence(
        self,
        reranked_documents
    ):
        """
        Determine whether the retrieved documents
        contain sufficient evidence to answer the query.

        The decision is based on the top Cross-Encoder
        reranker score.
        """

        if not reranked_documents:

            return {
                "status": "insufficient",
                "top_score": None,
                "relevant_chunks": 0
            }

        # -------------------------------------------------
        # Highest reranker score
        # -------------------------------------------------

        top_score = float(
            reranked_documents[0][
                "reranker_score"
            ]
        )

        # -------------------------------------------------
        # Count documents above threshold
        # -------------------------------------------------

        relevant_chunks = sum(
            1
            for document in reranked_documents
            if document["reranker_score"]
            >= self.EVIDENCE_THRESHOLD
        )

        # -------------------------------------------------
        # Evidence decision
        # -------------------------------------------------

        if top_score >= self.EVIDENCE_THRESHOLD:

            status = "sufficient"

        else:

            status = "insufficient"

        return {

            "status": status,

            "top_score": top_score,

            "relevant_chunks":
                relevant_chunks
        }

    # -----------------------------------------------------
    # Answer Query
    # -----------------------------------------------------

    def answer(
        self,
        query: str,
        top_k: int = 5
    ):
        """
        Execute the complete RAG pipeline:

        1. Query embedding
        2. Hybrid retrieval
        3. Cross-encoder reranking
        4. Evidence gate
        5. Context construction
        6. LLM generation
        """

        # -------------------------------------------------
        # 1. Query Embedding
        # -------------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        # -------------------------------------------------
        # 2. Hybrid Retrieval
        # -------------------------------------------------

        candidates = (
            self.hybrid_retriever.search(
                query=query,
                query_embedding=query_embedding,
                top_k=10,
                semantic_k=15,
                keyword_k=15
            )
        )

        # -------------------------------------------------
        # 3. Cross-Encoder Reranking
        # -------------------------------------------------

        reranked = (
            self.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=top_k
            )
        )

        # -------------------------------------------------
        # 4. Evidence Gate
        # -------------------------------------------------

        evidence = self.evaluate_evidence(
            reranked
        )

        print(
            "\nEvidence Gate: "
            f"{evidence['status'].upper()}"
        )

        print(
            f"Top reranker score: "
            f"{evidence['top_score']}"
        )

        print(
            f"Relevant chunks: "
            f"{evidence['relevant_chunks']}"
        )

        print(
            f"Required minimum score: "
            f"{self.EVIDENCE_THRESHOLD}"
        )

        # -------------------------------------------------
        # 5. REJECT if evidence is insufficient
        # -------------------------------------------------

        if evidence["status"] == "insufficient":

            answer = (
                "I couldn't find sufficient information "
                "in the provided documents to answer "
                "this question."
            )

            return {

                "answer": answer,

                "sources": reranked,

                "evidence": evidence
            }

        # -------------------------------------------------
        # 6. Build Context
        # -------------------------------------------------

        context = build_context(
            reranked
        )

        # -------------------------------------------------
        # 7. Generate Answer using LLM
        # -------------------------------------------------

        answer = self.llm.generate(
            query=query,
            context=context
        )

        # -------------------------------------------------
        # 8. Return Final Result
        # -------------------------------------------------

        return {

            "answer": answer,

            "sources": reranked,

            "evidence": evidence
        }