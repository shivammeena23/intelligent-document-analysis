from app.indexing.index_manager import IndexManager


index_manager = IndexManager()


print("=" * 80)
print("TESTING SAVED INDEX")
print("=" * 80)


if not index_manager.exists():

    print(
        "\nNo saved index found."
    )

else:

    index, chunks = (
        index_manager.load()
    )

    print(
        "\nIndex verification successful."
    )

    print(
        f"Number of vectors : {index.ntotal}"
    )

    print(
        f"Vector dimension  : {index.d}"
    )

    print(
        f"Number of chunks  : {len(chunks)}"
    )

    print("\nFirst chunk:")

    print(
        chunks[0]
    )