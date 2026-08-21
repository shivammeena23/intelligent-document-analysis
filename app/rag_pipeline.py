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


class RAGPipeline:

    def __init__(
        self,
        chunks,
        embeddings
    ):

        # ---------------------------------------------
        # Embedding model
        # ---------------------------------------------

        self.embedding_model = EmbeddingModel()


        # ---------------------------------------------
        # FAISS
        # ---------------------------------------------

        self.faiss_retriever = FAISSRetriever(
            dimension=embeddings.shape[1]
        )

        self.faiss_retriever.add_documents(
            embeddings,
            chunks
        )


        # ---------------------------------------------
        # BM25
        # ---------------------------------------------

        self.bm25_retriever = BM25Retriever(
            chunks
        )


        # ---------------------------------------------
        # Hybrid
        # ---------------------------------------------

        self.hybrid_retriever = HybridRetriever(
            faiss_retriever=self.faiss_retriever,
            bm25_retriever=self.bm25_retriever
        )


        # ---------------------------------------------
        # Reranker
        # ---------------------------------------------

        self.reranker = CrossEncoderReranker()


        # ---------------------------------------------
        # LLM
        # ---------------------------------------------

        self.llm = LLMGenerator()


    def answer(
        self,
        query: str,
        top_k: int = 5
    ):

        # ---------------------------------------------
        # 1. Embed query
        # ---------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )


        # ---------------------------------------------
        # 2. Hybrid retrieval
        # ---------------------------------------------

        candidates = (
            self.hybrid_retriever.search(
                query=query,
                query_embedding=query_embedding,
                top_k=10,
                semantic_k=15,
                keyword_k=15
            )
        )


        # ---------------------------------------------
        # 3. Reranking
        # ---------------------------------------------

        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k
        )


        # ---------------------------------------------
        # 4. Build context
        # ---------------------------------------------

        context = build_context(
            reranked
        )


        # ---------------------------------------------
        # 5. Generate answer
        # ---------------------------------------------

        answer = self.llm.generate(
            query=query,
            context=context
        )


        return {
            "answer": answer,
            "sources": reranked
        }