from app.query_engine import QueryEngine


print("=" * 80)
print("INTELLIGENT DOCUMENT ANALYSIS")
print("=" * 80)


engine = QueryEngine()


while True:

    query = input(
        "\nAsk a question "
        "(type 'exit' to quit): "
    )

    if query.lower().strip() == "exit":
        print("\nExiting...")
        break

    result = engine.answer(
        query=query,
        top_k=5
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(
        result["answer"]
    )

    # -----------------------------------------------------
    # Evidence information
    # -----------------------------------------------------

    evidence = result.get(
        "evidence",
        {}
    )

    print("\n" + "=" * 80)
    print("EVIDENCE")
    print("=" * 80)

    print(
        f"Status: "
        f"{evidence.get('status', 'unknown')}"
    )

    top_score = evidence.get(
        "top_score"
    )

    if top_score is not None:

        print(
            f"Top relevance score: "
            f"{top_score:.4f}"
        )

    print(
        f"Relevant chunks: "
        f"{evidence.get('relevant_chunks', 0)}"
    )

    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    if evidence.get("status") == "sufficient":

        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)

        for i, source in enumerate(
            result["sources"],
            start=1
        ):

            print(
                f"[{i}] "
                f"{source['source']} "
                f"(Page {source['page']})"
            )

    else:

        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)

        print(
            "No sufficiently relevant sources found."
        )