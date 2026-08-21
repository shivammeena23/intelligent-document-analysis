from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_documents


PDF_PATH = "data/documents/test.pdf"


documents = load_pdf(PDF_PATH)

chunks = chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Pages extracted: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

print("\n" + "=" * 80)

for chunk in chunks[:5]:

    print(f"Chunk ID : {chunk['chunk_id']}")
    print(f"Source   : {chunk['source']}")
    print(f"Page     : {chunk['page']}")
    print(f"Length   : {len(chunk['text'])}")

    print("\nText:")
    print(chunk["text"][:500])

    print("=" * 80)