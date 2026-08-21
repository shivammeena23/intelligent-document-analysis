from app.reranking.cross_encoder_reranker import (
    CrossEncoderReranker
)

from app.indexing.index_manager import (
    IndexManager
)

from app.embeddings.embedder import (
    EmbeddingModel
)

from app.retrieval.faiss_retriever import (
    FAISSRetriever
)

from app.retrieval.bm25_retriever import (
    BM25Retriever
)

from app.retrieval.hybrid_retriever import (
    HybridRetriever
)


# ---------------------------------------------------------
# Load saved index
# ---------------------------------------------------------

index_manager = IndexManager()

index, chunks = index_manager.load()


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

embedding_model = EmbeddingModel()


# ---------------------------------------------------------
# Setup FAISS
# ---------------------------------------------------------

faiss_retriever = FAISSRetriever(
    dimension=index.d
)

faiss_retriever.index = index

faiss_retriever.documents = chunks


# ---------------------------------------------------------
# Setup BM25
# ---------------------------------------------------------

bm25_retriever = BM25Retriever(
    chunks
)


# ---------------------------------------------------------
# Setup hybrid retriever
# ---------------------------------------------------------

hybrid_retriever = HybridRetriever(
    faiss_retriever=faiss_retriever,
    bm25_retriever=bm25_retriever
)


# ---------------------------------------------------------
# Setup reranker
# ---------------------------------------------------------

reranker = CrossEncoderReranker()


def inspect_query(query: str):

    print("\n")
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    # Query embedding

    query_embedding = embedding_model.encode(
        [query]
    )

    # Hybrid retrieval

    candidates = hybrid_retriever.search(
        query=query,
        query_embedding=query_embedding,
        top_k=10,
        semantic_k=15,
        keyword_k=15
    )

    # Reranking

    results = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=10
    )

    print("\nRERANKER SCORES")
    print("-" * 80)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n[{i}] "
            f"Score: {result['reranker_score']:.4f}"
        )

        print(
            f"Source: {result.get('source', 'Unknown')}"
        )

        print(
            f"Page: {result.get('page', 'Unknown')}"
        )

        print(
            f"Text: "
            f"{result.get('text', '')[:250]}"
        )


# ---------------------------------------------------------
# Relevant question
# ---------------------------------------------------------

inspect_query(
    "What information should be confirmed before construction begins?"
)


# ---------------------------------------------------------
# Irrelevant question
# ---------------------------------------------------------

inspect_query(
    "Who is the current CEO of NVIDIA?"
)