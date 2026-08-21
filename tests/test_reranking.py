from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_documents

from app.embeddings.embedder import EmbeddingModel

from app.retrieval.faiss_retriever import FAISSRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

from app.reranking.cross_encoder_reranker import (
    CrossEncoderReranker
)


PDF_PATH = "data/documents/test.pdf"


# ---------------------------------------------------------
# 1. Load PDF
# ---------------------------------------------------------

documents = load_pdf(PDF_PATH)

print(
    f"Pages extracted: {len(documents)}"
)


# ---------------------------------------------------------
# 2. Chunk documents
# ---------------------------------------------------------

chunks = chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

print(
    f"Chunks created: {len(chunks)}"
)


# ---------------------------------------------------------
# 3. Load embedding model
# ---------------------------------------------------------

embedding_model = EmbeddingModel()


# ---------------------------------------------------------
# 4. Create embeddings
# ---------------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts
)

print(
    f"Embedding shape: {embeddings.shape}"
)


# ---------------------------------------------------------
# 5. FAISS
# ---------------------------------------------------------

faiss_retriever = FAISSRetriever(
    dimension=embeddings.shape[1]
)

faiss_retriever.add_documents(
    embeddings,
    chunks
)


# ---------------------------------------------------------
# 6. BM25
# ---------------------------------------------------------

bm25_retriever = BM25Retriever(
    documents=chunks
)


# ---------------------------------------------------------
# 7. Hybrid Retriever
# ---------------------------------------------------------

hybrid_retriever = HybridRetriever(
    faiss_retriever=faiss_retriever,
    bm25_retriever=bm25_retriever
)


# ---------------------------------------------------------
# 8. Reranker
# ---------------------------------------------------------

reranker = CrossEncoderReranker()


# ---------------------------------------------------------
# 9. Query
# ---------------------------------------------------------

query = input(
    "\nAsk a question about the document: "
)


# ---------------------------------------------------------
# 10. Query embedding
# ---------------------------------------------------------

query_embedding = embedding_model.encode(
    [query]
)


# ---------------------------------------------------------
# 11. Hybrid retrieval
# ---------------------------------------------------------

candidate_documents = hybrid_retriever.search(
    query=query,
    query_embedding=query_embedding,
    top_k=10,
    semantic_k=15,
    keyword_k=15
)


print(
    f"\nHybrid candidates: "
    f"{len(candidate_documents)}"
)


# ---------------------------------------------------------
# 12. Reranking
# ---------------------------------------------------------

results = reranker.rerank(
    query=query,
    documents=candidate_documents,
    top_k=5
)


# ---------------------------------------------------------
# 13. Display results
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("RERANKED RESULTS")
print("=" * 80)


for i, result in enumerate(
    results,
    start=1
):

    print(
        f"\nResult {i}"
    )

    print("-" * 80)

    print(
        f"Reranker Score : "
        f"{result['reranker_score']:.4f}"
    )

    print(
        f"Fusion Score   : "
        f"{result['fusion_score']:.6f}"
    )

    print(
        f"Source         : "
        f"{result['source']}"
    )

    print(
        f"Page           : "
        f"{result['page']}"
    )

    print(
        f"Chunk          : "
        f"{result['chunk_id']}"
    )

    print("\nText:")

    print(
        result["text"][:700]
    )

    print("=" * 80)