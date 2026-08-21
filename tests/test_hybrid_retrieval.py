from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.embeddings.embedder import EmbeddingModel
from app.retrieval.faiss_retriever import FAISSRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever


PDF_PATH = "data/documents/test.pdf"


# ---------------------------------------------------------
# 1. Load PDF
# ---------------------------------------------------------

documents = load_pdf(PDF_PATH)

print(f"Pages extracted: {len(documents)}")


# ---------------------------------------------------------
# 2. Chunk documents
# ---------------------------------------------------------

chunks = chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Chunks created: {len(chunks)}")


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

embeddings = embedding_model.encode(texts)

print(f"Embedding shape: {embeddings.shape}")


# ---------------------------------------------------------
# 5. FAISS retriever
# ---------------------------------------------------------

faiss_retriever = FAISSRetriever(
    dimension=embeddings.shape[1]
)

faiss_retriever.add_documents(
    embeddings,
    chunks
)


# ---------------------------------------------------------
# 6. BM25 retriever
# ---------------------------------------------------------

bm25_retriever = BM25Retriever(
    documents=chunks
)


# ---------------------------------------------------------
# 7. Hybrid retriever
# ---------------------------------------------------------

hybrid_retriever = HybridRetriever(
    faiss_retriever=faiss_retriever,
    bm25_retriever=bm25_retriever
)


# ---------------------------------------------------------
# 8. Query
# ---------------------------------------------------------

query = input(
    "\nAsk a question about the document: "
)


# ---------------------------------------------------------
# 9. Query embedding
# ---------------------------------------------------------

query_embedding = embedding_model.encode(
    [query]
)


# ---------------------------------------------------------
# 10. Hybrid search
# ---------------------------------------------------------

results = hybrid_retriever.search(
    query=query,
    query_embedding=query_embedding,
    top_k=5
)


# ---------------------------------------------------------
# 11. Display results
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("HYBRID SEARCH RESULTS")
print("=" * 80)

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("-" * 80)

    print(
        f"Fusion Score  : "
        f"{result['fusion_score']:.6f}"
    )

    print(
        f"Semantic Rank : "
        f"{result['semantic_rank']}"
    )

    print(
        f"Keyword Rank  : "
        f"{result['keyword_rank']}"
    )

    print(f"Source        : {result['source']}")
    print(f"Page          : {result['page']}")
    print(f"Chunk         : {result['chunk_id']}")

    print("\nText:")
    print(result["text"][:700])

    print("=" * 80)