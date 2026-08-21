import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.query_engine import QueryEngine


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Intelligent Document Analysis",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("📄 Intelligent Document Analysis")

st.write(
    "Ask questions about the documents in the knowledge base."
)


# ---------------------------------------------------------
# Load Query Engine
# ---------------------------------------------------------

@st.cache_resource
def load_engine():

    return QueryEngine()


with st.spinner("Loading document analysis system..."):

    engine = load_engine()


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

query = st.text_input(
    "Ask a question",
    placeholder="e.g. What information is required from the site survey before construction?"
)


# ---------------------------------------------------------
# Ask button
# ---------------------------------------------------------

if st.button("Ask Question"):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner("Analyzing documents..."):

            result = engine.answer(
                query=query,
                top_k=5
            )


        # -------------------------------------------------
        # Answer
        # -------------------------------------------------

        st.subheader("Answer")

        st.write(
            result.get(
                "answer",
                "No answer generated."
            )
        )


        # -------------------------------------------------
        # Evidence
        # -------------------------------------------------

        evidence = result.get(
            "evidence",
            {}
        )

        if evidence:

            st.subheader("Evidence")

            status = evidence.get(
                "status",
                "unknown"
            )

            if status == "sufficient":

                st.success(
                    "Sufficient evidence found in the documents."
                )

            else:

                st.warning(
                    "Insufficient evidence found in the documents."
                )


        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        sources = result.get(
            "sources",
            []
        )

        st.subheader("Sources")

        if sources:

            for i, source in enumerate(
                sources,
                start=1
            ):

                document = source.get(
                    "source",
                    "Unknown"
                )

                page = source.get(
                    "page",
                    "Unknown"
                )

                score = source.get(
                    "reranker_score"
                )

                st.markdown(
                    f"**[{i}] {document} — Page {page}**"
                )

                if score is not None:

                    st.caption(
                        f"Reranker score: {score:.4f}"
                    )

        else:

            st.write(
                "No sources available."
            )