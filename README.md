# Intelligent Document Analysis

A retrieval-augmented generation (RAG) system for asking questions about a collection of PDF documents and receiving grounded answers with source citations.

The system retrieves relevant document chunks, reranks them, applies an evidence gate, and uses Gemini to generate an answer only when sufficient evidence is available.

## Key Features

- PDF document ingestion and text extraction
- Chunking of document content for retrieval
- FAISS-based semantic retrieval
- BM25 keyword retrieval
- Hybrid retrieval combining semantic and lexical search
- Cross-encoder reranking
- Evidence sufficiency gate to reduce unsupported answers
- Gemini-based answer generation
- Source/page citations in generated answers
- Streamlit web interface
- Automated evaluation with accuracy, precision, recall, F1, false-accept rate, false-reject rate, and latency
- Retry handling for temporary Gemini API failures

## System Architecture

```text
PDF Documents
      │
      ▼
Document Loading
      │
      ▼
Text Chunking
      │
      ▼
Embeddings ───────────────┐
      │                   │
      ▼                   ▼
   FAISS              BM25 Retrieval
      │                   │
      └─────────┬─────────┘
                ▼
         Hybrid Retrieval
                │
                ▼
        Cross-Encoder Reranker
                │
                ▼
          Evidence Gate
          /           \
   Sufficient        Insufficient
       │                  │
       ▼                  ▼
 Gemini Generation   Safe fallback
       │
       ▼
 Answer + Citations
```

## Tech Stack

- **Python**
- **Streamlit** — web interface
- **FAISS** — vector similarity search
- **BM25** — lexical retrieval
- **Sentence Transformers** — embeddings and reranking
- **Google Gemini API** — answer generation
- **PyPDF / PDF loaders** — document ingestion
- **JSON** — evaluation dataset and results

## Project Structure

```text
intelligent-document-analysis/
│
├── app/
│   ├── embeddings/
│   ├── generation/
│   │   ├── context_builder.py
│   │   └── llm.py
│   ├── indexing/
│   ├── ingestion/
│   ├── reranking/
│   ├── retrieval/
│   ├── query_engine.py
│   └── rag_pipeline.py
│
├── data/
│   └── documents/
│
├── evaluation/
│   ├── evaluator.py
│   ├── questions.json
│   ├── results.json
│   └── threshold_analysis.py
│
├── frontend/
│   └── app.py
│
├── tests/
│   ├── test_chunking.py
│   ├── test_document_loader.py
│   ├── test_hybrid_retrieval.py
│   ├── test_index_manager.py
│   ├── test_llm.py
│   ├── test_query_engine.py
│   ├── test_rag.py
│   ├── test_reranker_scores.py
│   ├── test_reranking.py
│   └── test_retrieval.py
│
├── .gitignore
└── README.md
```

## How the RAG Pipeline Works

### 1. Document Ingestion

PDF files are loaded from the document collection and converted into searchable text.

### 2. Chunking

Documents are divided into smaller chunks so that retrieval can identify the most relevant sections rather than passing entire documents to the language model.

### 3. Hybrid Retrieval

Two complementary retrieval approaches are used:

- **FAISS semantic retrieval** finds conceptually similar content.
- **BM25 retrieval** finds relevant keyword-based matches.

Their results are combined to improve retrieval coverage.

### 4. Reranking

Retrieved chunks are passed through a cross-encoder reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This provides a more precise relevance score for the query-document pairs.

### 5. Evidence Gate

The system checks the reranked evidence before generation.

If the evidence is sufficient, the question proceeds to Gemini.

If the evidence is insufficient, the system avoids generating an unsupported answer and returns a safe fallback response.

### 6. Answer Generation

Gemini generates the final answer using only the retrieved document context.

The generation prompt requires factual statements to contain source citations such as:

```text
[1]
[2]
[3]
```

This keeps the answer grounded in the retrieved documents.

## Evaluation

The project includes an evaluation pipeline using a labelled set of 20 questions:

- 10 answerable questions
- 10 unanswerable questions

The latest successful evaluation produced:

| Metric | Result |
|---|---:|
| Total Questions | 20 |
| Evaluated Questions | 20 |
| True Positive | 10 |
| True Negative | 10 |
| False Positive | 0 |
| False Negative | 0 |
| Accuracy | 1.00 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 Score | 1.00 |
| False Accept Rate | 0.00 |
| False Reject Rate | 0.00 |
| Average Latency | ~6.89 s |

The evaluation results are stored in:

```text
evaluation/results.json
```

> Note: latency and API availability can vary depending on the Gemini API service and local machine.

## Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/shivammeena23/intelligent-document-analysis.git
cd intelligent-document-analysis
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git through `.gitignore`.

### 4. Run the Streamlit application

```powershell
streamlit run frontend/app.py --server.fileWatcherType none
```

The application will be available at:

```text
http://localhost:8501
```

## Running the Evaluation

From the project root:

```powershell
python -m evaluation.evaluator
```

The evaluation script reads questions from:

```text
evaluation/questions.json
```

and writes the results to:

```text
evaluation/results.json
```

## Example

A question can be entered through the Streamlit interface, for example:

```text
What information should be confirmed before construction begins?
```

For questions supported by the uploaded documents, the system returns a grounded answer followed by the relevant document sources.

For questions outside the document knowledge base, the system returns a safe insufficient-evidence response instead of relying on general knowledge.

## Testing

The project includes unit/integration tests covering:

- document loading
- chunking
- retrieval
- hybrid retrieval
- indexing
- reranking
- LLM generation
- RAG pipeline
- query engine

Run the test suite with:

```powershell
python -m pytest
```

## Design Goals

The project focuses on three important RAG properties:

1. **Retrieval quality** — find relevant document evidence.
2. **Grounded generation** — answer using retrieved evidence rather than outside knowledge.
3. **Safe refusal** — avoid answering when the available document evidence is insufficient.

## Future Improvements

Possible extensions include:

- richer document-format support
- improved citation rendering
- conversation history
- larger evaluation datasets
- deployment to a cloud platform
- additional retrieval and reranking experiments

## Author

**Shivam Meena**

IIT Kanpur

GitHub: `https://github.com/shivammeena23`
