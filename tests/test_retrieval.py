from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.embeddings.embedder import EmbeddingModel
from app.retrieval.faiss_retriever import FAISSRetriever


PDF_PATH = "data/documents/test.pdf"


# ---------------------------------------------------------
# 1. Load PDF
# ---------------------------------------------------------

documents = load_pdf(PDF_PATH)

print(f"\nPages extracted: {len(documents)}")


# ---------------------------------------------------------
# 2. Create chunks
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
# 4. Create embeddings for chunks
# ---------------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(texts)

print(
    f"\nEmbedding shape: {embeddings.shape}"
)


# ---------------------------------------------------------
# 5. Create FAISS index
# ---------------------------------------------------------

embedding_dimension = embeddings.shape[1]

retriever = FAISSRetriever(
    dimension=embedding_dimension
)


# ---------------------------------------------------------
# 6. Add chunks to FAISS
# ---------------------------------------------------------

retriever.add_documents(
    embeddings,
    chunks
)

print(
    f"FAISS index contains "
    f"{retriever.index.ntotal} vectors."
)


# ---------------------------------------------------------
# 7. Ask a question
# ---------------------------------------------------------

query = input(
    "\nAsk a question about the document: "
)


# ---------------------------------------------------------
# 8. Embed the question
# ---------------------------------------------------------

query_embedding = embedding_model.encode(
    [query]
)


# ---------------------------------------------------------
# 9. Search FAISS
# ---------------------------------------------------------

results = retriever.search(
    query_embedding,
    top_k=5
)


# ---------------------------------------------------------
# 10. Display results
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("TOP RETRIEVED RESULTS")
print("=" * 80)

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("-" * 80)

    print(f"Score  : {result['score']:.4f}")
    print(f"Source : {result['source']}")
    print(f"Page   : {result['page']}")
    print(f"Chunk  : {result['chunk_id']}")

    print("\nText:")
    print(result["text"][:700])

    print("=" * 80)