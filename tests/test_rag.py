from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_documents

from app.embeddings.embedder import EmbeddingModel

from app.rag_pipeline import RAGPipeline


PDF_PATH = "data/documents/test.pdf"


# ---------------------------------------------------------
# 1. Load PDF
# ---------------------------------------------------------

documents = load_pdf(
    PDF_PATH
)

print(
    f"Pages extracted: {len(documents)}"
)


# ---------------------------------------------------------
# 2. Chunk
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
# 3. Embeddings
# ---------------------------------------------------------

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts
)


# ---------------------------------------------------------
# 4. Create RAG pipeline
# ---------------------------------------------------------

rag = RAGPipeline(
    chunks=chunks,
    embeddings=embeddings
)


# ---------------------------------------------------------
# 5. Ask question
# ---------------------------------------------------------

query = input(
    "\nAsk a question about the document: "
)


# ---------------------------------------------------------
# 6. Generate answer
# ---------------------------------------------------------

result = rag.answer(
    query=query
)


# ---------------------------------------------------------
# 7. Display answer
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(
    result["answer"]
)


# ---------------------------------------------------------
# 8. Display sources
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("SOURCES")
print("=" * 80)


for source in result["sources"]:

    print(
        f"- {source['source']} "
        f"(Page {source['page']})"
    )