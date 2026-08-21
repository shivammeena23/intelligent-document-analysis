from app.ingestion.document_loader import load_documents


documents = load_documents(
    "data/documents"
)


print("\n" + "=" * 80)
print("DOCUMENT LOADER TEST")
print("=" * 80)


sources = {}

for document in documents:

    source = document["source"]

    sources[source] = (
        sources.get(source, 0) + 1
    )


print("\nDocuments found:")

for source, page_count in sources.items():

    print(
        f"- {source}: "
        f"{page_count} pages"
    )


print(
    f"\nTotal pages: {len(documents)}"
)