# Insurellm AI Assistant 🤖

An intelligent RAG (Retrieval-Augmented Generation) chatbot for insurance domain knowledge, powered by LLMs and vector search.

## Overview

Insurellm AI Assistant is a Streamlit-based application that provides intelligent responses to insurance-related queries using RAG architecture. It leverages a comprehensive knowledge base containing company information, product details, contracts, and employee data.

## Features

- **RAG-Powered Responses**: Combines vector search with LLM generation for accurate, contextual answers
- **Document Reranking**: AI-powered chunk reranking for improved relevance
- **Interactive Chat Interface**: Clean Streamlit UI with conversation history
- **Comprehensive Knowledge Base**: Covers 8 insurance product lines and 32+ contracts

## Products Covered

| Product | Description |
|---------|-------------|
| **Markellm** | Insurance marketplace connecting consumers with providers |
| **Carllm** | Auto insurance portal |
| **Homellm** | Home insurance portal |
| **Rellm** | Enterprise reinsurance platform |
| **Lifellm** | Life insurance solutions |
| **Healthllm** | Health insurance platform |
| **Bizllm** | Commercial insurance |
| **Claimllm** | Claims processing system |

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Ollama (Llama 3.2)
- **Vector Database**: ChromaDB
- **Embeddings**: Nomic Embed Text
- **Framework**: LiteLLM, OpenAI SDK, Pydantic

## Project Structure

```
insurellm_project/
├── knowledge-base/
│   ├── company/          # Company info (about, careers, culture)
│   ├── products/         # Product documentation (8 products)
│   ├── contracts/        # Client contracts (32+ contracts)
│   └── employees/        # Employee profiles
├── src/
│   ├── rag_app.py                # Main Streamlit application
│   ├── day_5_rac.ipynb           # Development notebook
│   ├── requirements_streamlit.txt # Dependencies
│   └── pre_processed1_db/        # ChromaDB vector store
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ahmedpasha6666/insurellm-ai-assistant.git
   cd insurellm-ai-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r src/requirements_streamlit.txt
   ```

3. **Pull required Ollama models**
   ```bash
   ollama pull llama3.2:latest
   ollama pull nomic-embed-text
   ```

4. **Run the application**
   ```bash
   streamlit run src/rag_app.py
   ```

## Usage

1. Start the Streamlit app
2. Enter your insurance-related question in the chat input
3. The system will:
   - Search the knowledge base for relevant documents
   - Rerank results by relevance
   - Generate a contextual response using the LLM

## Configuration

Key configuration options in `rag_app.py`:

```python
MODEL = "ollama/llama3.2:latest"      # LLM model
EMBEDDING_MODEL = "nomic-embed-text"   # Embedding model
COLLECTION_NAME = "docs"               # ChromaDB collection
```

## Requirements

```
streamlit>=1.28.0
openai>=1.0.0
chromadb>=0.4.0
litellm>=1.0.0
pydantic>=2.0.0
tqdm
numpy
sklearn
plotly
```

## License

This project is for educational and demonstration purposes.

## Author

Ahmed Pasha

---

*Built with using RAG architecture and Ollama*
