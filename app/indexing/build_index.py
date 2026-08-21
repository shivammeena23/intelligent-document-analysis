from app.ingestion.document_loader import (
    load_documents
)

from app.ingestion.chunker import (
    chunk_documents
)

from app.embeddings.embedder import (
    EmbeddingModel
)

from app.retrieval.faiss_retriever import (
    FAISSRetriever
)

from app.indexing.index_manager import (
    IndexManager
)


DOCUMENT_DIRECTORY = "data/documents"


def build_index():

    print("=" * 80)
    print("MULTI-DOCUMENT INDEXING")
    print("=" * 80)

    # -----------------------------------------------------
    # 1. Load all PDFs
    # -----------------------------------------------------

    print("\n[1/5] Loading documents...")

    documents = load_documents(
        DOCUMENT_DIRECTORY
    )

    # -----------------------------------------------------
    # 2. Create chunks
    # -----------------------------------------------------

    print("\n[2/5] Creating chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    # -----------------------------------------------------
    # 3. Generate embeddings
    # -----------------------------------------------------

    print("\n[3/5] Generating embeddings...")

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # -----------------------------------------------------
    # 4. Build FAISS
    # -----------------------------------------------------

    print("\n[4/5] Building FAISS index...")

    retriever = FAISSRetriever(
        dimension=embeddings.shape[1]
    )

    retriever.add_documents(
        embeddings,
        chunks
    )

    print(
        f"Vectors indexed: "
        f"{retriever.index.ntotal}"
    )

    # -----------------------------------------------------
    # 5. Save index
    # -----------------------------------------------------

    print("\n[5/5] Saving index...")

    index_manager = IndexManager()

    index_manager.save(
        index=retriever.index,
        chunks=chunks
    )

    print("\n" + "=" * 80)
    print("MULTI-DOCUMENT INDEXING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    build_index()