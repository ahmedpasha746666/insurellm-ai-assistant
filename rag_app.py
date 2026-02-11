"""
================================================================================
                        INSURELLM RAG AI ASSISTANT
================================================================================

Author: Ahmed Pasha
Version: 1.0
Description: A Retrieval-Augmented Generation (RAG) chatbot for insurance domain

================================================================================
                              HOW RAG WORKS
================================================================================

RAG (Retrieval-Augmented Generation) combines the power of:
1. Information Retrieval (searching relevant documents)
2. Large Language Models (generating human-like responses)

              ┌─────────────────────────────────────────────────────────┐
              │                    RAG PIPELINE                          │
              └─────────────────────────────────────────────────────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   1. USER QUESTION    │
                          │   "What is Markellm?" │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   2. QUERY REWRITING  │
                          │   Refine the question │
                          │   for better search   │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   3. EMBEDDING        │
                          │   Convert query to    │
                          │   768-dim vector      │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   4. VECTOR SEARCH    │
                          │   Find similar docs   │
                          │   in ChromaDB         │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   5. RERANKING        │
                          │   AI reorders chunks  │
                          │   by relevance        │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   6. CONTEXT BUILDING │
                          │   Combine top chunks  │
                          │   into prompt         │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   7. LLM GENERATION   │
                          │   Generate answer     │
                          │   with context        │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │   8. RESPONSE         │
                          │   Return to user      │
                          └───────────────────────┘

================================================================================
                              MODELS USED
================================================================================

1. LLM (Large Language Model):
   - Model: Llama 3.2 (Latest) via Ollama
   - Purpose: Query rewriting, reranking, answer generation
   - Parameters: 3B parameters
   - Context: 128K tokens

2. Embedding Model:
   - Model: nomic-embed-text via Ollama
   - Purpose: Convert text to 768-dimensional vectors
   - Embedding Dimension: 768
   - Max Tokens: 8192

================================================================================
                              DATA INFORMATION
================================================================================

Knowledge Base Structure:
├── company/          → Company info (about, careers, culture, overview)
├── products/         → 8 Insurance Products:
│   ├── Markellm      - Insurance marketplace
│   ├── Carllm        - Auto insurance portal
│   ├── Homellm       - Home insurance portal
│   ├── Rellm         - Enterprise reinsurance
│   ├── Lifellm       - Life insurance
│   ├── Healthllm     - Health insurance
│   ├── Bizllm        - Commercial insurance
│   └── Claimllm      - Claims processing
├── contracts/        → 32+ Client contracts
└── employees/        → Employee profiles

Vector Database:
- Database: ChromaDB (Persistent)
- Collection: "docs"
- Chunks: ~500+ document chunks
- Chunk Size: ~500 tokens with overlap

================================================================================
                              TECH STACK
================================================================================

- Frontend:     Streamlit (Interactive Web UI)
- LLM Backend:  Ollama (Local LLM server)
- Vector DB:    ChromaDB (Persistent vector storage)
- Embeddings:   nomic-embed-text (via Ollama)
- Framework:    LiteLLM (Unified LLM API)
- Validation:   Pydantic (Data validation)

================================================================================
"""

import streamlit as st
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from litellm import completion
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Insurellm AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================
# LLM Model: Llama 3.2 running locally via Ollama
MODEL = "ollama/llama3.2:latest"

# ChromaDB path containing pre-processed document embeddings
DB_NAME = r"./pre_processed1_db"
# Embedding model: nomic-embed-text (768-dimensional vectors)
EMBEDDING_MODEL = "nomic-embed-text"

# ChromaDB collection name where document chunks are stored
COLLECTION_NAME = "docs"

# ============================================================================
# DATABASE CONNECTION
# ============================================================================
# Initialize OpenAI-compatible client for embeddings (via Ollama)
@st.cache_resource
def get_openai_client():
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Initialize ChromaDB - Persistent vector database for document storage
@st.cache_resource
def get_collection():
    """Connect to ChromaDB and return the document collection."""
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    return collection

# ============================================================================
# DATA MODELS (Pydantic)
# ============================================================================
class Result(BaseModel):
    page_content: str
    metadata: dict

class RankOrder(BaseModel):
    """Schema for reranking response - list of chunk IDs ordered by relevance."""
    order: list[int] = Field(
        description="The order of relevance of chunks, from most to least relevant, by chunk id number"
    )

# ============================================================================
# RAG PIPELINE FUNCTIONS
# ============================================================================

def fetch_context_unranked(question, collection, openai_client, k=10):
    """
    STEP 1: VECTOR RETRIEVAL
    
    Converts the question into a vector embedding and performs similarity
    search in ChromaDB to find the K most relevant document chunks.
    
    Args:
        question: User's query string
        collection: ChromaDB collection
        openai_client: Client for embedding generation
        k: Number of chunks to retrieve (default: 10)
    
    Returns:
        List of Result objects containing page_content and metadata
    """
    query = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, 
        input=[question]
    ).data[0].embedding
    
    results = collection.query(query_embeddings=[query], n_results=k)
    
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    
    return chunks

def rerank(question, chunks):
    """
    STEP 2: AI-POWERED RERANKING
    
    Uses the LLM to reorder retrieved chunks by semantic relevance.
    This improves answer quality by prioritizing the most relevant content.
    
    Args:
        question: User's query string
        chunks: List of retrieved document chunks
    
    Returns:
        Reordered list of chunks (most relevant first)
    """
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids.
"""
    
    user_prompt = f"The user has asked: {question}\n\nOrder all chunks by relevance.\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = completion(
        model=MODEL,
        messages=messages,
        response_format=RankOrder,
        timeout=300
    )
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    
    return [chunks[i - 1] for i in order]

def rewrite_query(question, history=[]):
    """
    STEP 3: QUERY REWRITING
    
    Refines the user's question to improve retrieval accuracy.
    Converts conversational queries into focused search queries.
    
    Args:
        question: Original user question
        history: Conversation history for context
    
    Returns:
        Rewritten, more specific query for knowledge base search
    """
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base.

This is the conversation history:
{history}

And this is the user's current question:
{question}

Respond with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content.
Focus on the question details. Don't mention the company name unless it's a general question.
IMPORTANT: Respond ONLY with the knowledge base query, nothing else.
"""
    
    response = completion(
        model=MODEL,
        messages=[{"role": "system", "content": message}],
        timeout=120
    )
    return response.choices[0].message.content

def answer_question(question, collection, openai_client, k=15, use_reranking=True):
    """
    COMPLETE RAG PIPELINE
    
    Orchestrates the full RAG workflow:
    1. Rewrite query for better retrieval
    2. Retrieve relevant chunks via vector search
    3. Rerank chunks using AI (optional)
    4. Build context from top chunks
    5. Generate answer using LLM with context
    
    Args:
        question: User's question
        collection: ChromaDB collection
        openai_client: Embedding client
        k: Number of chunks to retrieve
        use_reranking: Whether to apply AI reranking
    
    Returns:
        Tuple of (answer, chunks, rewritten_query)
    """
    # Step 1: Rewrite query for better retrieval
    rewritten_query = rewrite_query(question)
    
    # Step 2: Retrieve chunks
    chunks = fetch_context_unranked(rewritten_query, collection, openai_client, k)
    
    # Step 3: Rerank (optional)
    if use_reranking and len(chunks) > 0:
        chunks = rerank(rewritten_query, chunks)
    
    # Step 4: Build context
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" 
        for chunk in chunks
    )
    
    system_prompt = f"""
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness.
If you don't know the answer, say so.

For context, here are specific extracts from the Knowledge Base that might be directly relevant:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Step 5: Get answer
    response = completion(
        model=MODEL,
        messages=messages,
        timeout=300
    )
    
    return response.choices[0].message.content, chunks, rewritten_query

# ============================================================================
# STREAMLIT USER INTERFACE
# ============================================================================

def main():
    """Main Streamlit application entry point."""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .step-box {
        background: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    # Initialize chat visibility state
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Check database status early
    try:
        collection = get_collection()
        doc_count = collection.count()
        db_status = True
    except Exception as e:
        doc_count = 0
        db_status = False
    
    # =========================================================================
    # CHAT PAGE (Separate clean view)
    # =========================================================================
    
    if st.session_state.show_chat:
        # Chat Page Header
        st.title("💬 Chat with Insurellm AI")
        st.markdown("*Ask questions about products, contracts, employees, and company information*")
        
        st.divider()
        
        # Sidebar for chat settings
        with st.sidebar:
            st.header("⚙️ Chat Settings")
            
            if db_status:
                st.success(f"✅ Database: {doc_count} chunks")
            else:
                st.error("❌ Database not found")
                st.stop()
            
            st.divider()
            
            retrieval_k = st.slider(
                "Chunks to retrieve",
                min_value=5,
                max_value=30,
                value=15
            )
            
            use_reranking = st.checkbox("Use AI Reranking", value=True)
            show_sources = st.checkbox("Show sources", value=True)
            
            st.divider()
            st.markdown("### 💡 Try asking:")
            st.markdown("""
            - What products does Insurellm offer?
            - Tell me about Markellm
            - Who is the CEO?
            - How many employees work remotely?
            """)
            
            st.divider()
            
            # Back to Home button in sidebar
            if st.button("🏠 Back to Home", use_container_width=True):
                st.session_state.show_chat = False
                st.rerun()
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    if message["role"] == "assistant" and "sources" in message and show_sources:
                        with st.expander("📄 View Sources"):
                            for i, source in enumerate(message["sources"][:3], 1):
                                st.markdown(f"**{i}. {source['file']}**")
                                st.text(source['preview'])
                                st.divider()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about Insurellm..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    try:
                        openai_client = get_openai_client()
                        collection = get_collection()
                        
                        status_placeholder = st.empty()
                        status_placeholder.info("🔍 Searching knowledge base...")
                        
                        answer, chunks, rewritten_query = answer_question(
                            prompt,
                            collection,
                            openai_client,
                            k=retrieval_k,
                            use_reranking=use_reranking
                        )
                        
                        status_placeholder.success(f"✨ Found {len(chunks)} relevant documents")
                        time.sleep(1)
                        status_placeholder.empty()
                        
                        st.markdown(answer)
                        
                        sources = []
                        for chunk in chunks[:3]:
                            sources.append({
                                "file": chunk.metadata.get('source', 'Unknown').split('/')[-1],
                                "preview": chunk.page_content[:200] + "..."
                            })
                        
                        if show_sources:
                            with st.expander("📄 View Sources"):
                                st.markdown(f"**Rewritten Query:** _{rewritten_query}_")
                                st.divider()
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**{i}. {source['file']}**")
                                    st.text(source['preview'])
                                    st.divider()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.error("Make sure Ollama is running: `ollama serve`")
        
        # Bottom buttons - Delete History on left
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("🗑️ Delete History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # =========================================================================
    # HOME PAGE (Info sections)
    # =========================================================================
    
    else:
        # Header
        st.title("🤖 Insurellm AI Assistant")
        st.markdown("##### *Powered by RAG (Retrieval-Augmented Generation) | Built with Llama 3.2 & ChromaDB*")
        
        # Chat Button - Centered at top
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💬 Start Chatting with AI", use_container_width=True, type="primary"):
                st.session_state.show_chat = True
                st.rerun()
        
        st.divider()
        
        # Top Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Knowledge Chunks", f"{doc_count}+")
        with col2:
            st.metric("📦 Products", "8")
        with col3:
            st.metric("📄 Contracts", "32+")
        with col4:
            st.metric("🧠 Model", "Llama 3.2")
        
        st.divider()
        
        # Info Sections - Collapsed by default (click to expand)
        st.markdown("### 📖 Learn More About This System")
        
        with st.expander("🔄 How RAG Works", expanded=False):
            st.markdown("""
            **RAG (Retrieval-Augmented Generation)** is an AI architecture that enhances LLM responses 
            by retrieving relevant information from a knowledge base before generating answers.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Pipeline Steps:")
                steps = [
                    ("1️⃣", "User Question", "You ask a question about Insurellm"),
                    ("2️⃣", "Query Rewriting", "AI refines your question for better search"),
                    ("3️⃣", "Embedding", "Question converted to 768-dimensional vector"),
                    ("4️⃣", "Vector Search", "Find similar documents in ChromaDB"),
                ]
                for icon, title, desc in steps:
                    st.markdown(f"**{icon} {title}**")
                    st.caption(desc)
            
            with col2:
                st.markdown("#### &nbsp;")
                steps = [
                    ("5️⃣", "Reranking", "AI reorders chunks by relevance"),
                    ("6️⃣", "Context Building", "Top chunks combined into prompt"),
                    ("7️⃣", "LLM Generation", "Llama 3.2 generates contextual answer"),
                    ("8️⃣", "Response", "Answer returned with source references"),
                ]
                for icon, title, desc in steps:
                    st.markdown(f"**{icon} {title}**")
                    st.caption(desc)
            
            st.info("💡 **Why RAG?** Unlike traditional chatbots, RAG provides accurate, up-to-date answers by grounding responses in your actual documents rather than relying solely on pre-trained knowledge.")
        
        with st.expander("🧠 Models Used", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🤖 Large Language Model (LLM)")
                st.markdown("""
                | Property | Value |
                |----------|-------|
                | **Model** | Llama 3.2 (Latest) |
                | **Provider** | Ollama (Local) |
                | **Parameters** | 3 Billion |
                | **Context Window** | 128K tokens |
                | **Purpose** | Query rewriting, Reranking, Answer generation |
                """)
            
            with col2:
                st.markdown("#### 📐 Embedding Model")
                st.markdown("""
                | Property | Value |
                |----------|-------|
                | **Model** | nomic-embed-text |
                | **Provider** | Ollama (Local) |
                | **Dimensions** | 768 |
                | **Max Tokens** | 8,192 |
                | **Purpose** | Convert text to vector embeddings |
                """)
            
            st.success("✅ All models run **locally** via Ollama - no data leaves your machine!")
        
        with st.expander("📊 Knowledge Base", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📁 Document Categories")
                st.markdown("""
                | Category | Contents |
                |----------|----------|
                | 🏢 **Company** | About, Careers, Culture, Overview |
                | 📦 **Products** | 8 Insurance product lines |
                | 📄 **Contracts** | 32+ client contracts |
                | 👥 **Employees** | Team member profiles |
                """)
            
            with col2:
                st.markdown("#### 📦 Insurance Products")
                products = [
                    ("🏪 Markellm", "Insurance marketplace"),
                    ("🚗 Carllm", "Auto insurance portal"),
                    ("🏠 Homellm", "Home insurance portal"),
                    ("🔄 Rellm", "Enterprise reinsurance"),
                    ("❤️ Lifellm", "Life insurance"),
                    ("🏥 Healthllm", "Health insurance"),
                    ("💼 Bizllm", "Commercial insurance"),
                    ("📋 Claimllm", "Claims processing"),
                ]
                for name, desc in products:
                    st.markdown(f"- **{name}** - {desc}")
            
            st.markdown("#### 🗄️ Vector Database")
            col1, col2, col3 = st.columns(3)
            col1.metric("Database", "ChromaDB")
            col2.metric("Collection", "docs")
            col3.metric("Chunk Size", "~500 tokens")
        
        with st.expander("🛠️ Tech Stack", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Core Technologies")
                st.markdown("""
                | Component | Technology |
                |-----------|------------|
                | 🎨 **Frontend** | Streamlit |
                | 🤖 **LLM Backend** | Ollama (Local) |
                | 🗄️ **Vector DB** | ChromaDB |
                | 📐 **Embeddings** | nomic-embed-text |
                | 🔗 **LLM Framework** | LiteLLM |
                | ✅ **Validation** | Pydantic |
                """)
            
            with col2:
                st.markdown("#### Architecture Benefits")
                st.markdown("""
                - 🔒 **Privacy**: All processing runs locally
                - ⚡ **Speed**: No API latency
                - 💰 **Cost**: Free to run (no API costs)
                - 🎯 **Accuracy**: Grounded in your documents
                - 🔄 **Flexible**: Easy to update knowledge base
                """)
            
            st.code("""
# Quick Start Commands
ollama pull llama3.2:latest      # Download LLM
ollama pull nomic-embed-text     # Download embeddings
streamlit run rag_app.py         # Start the app
            """, language="bash")
        
        # Sidebar for Home Page
        with st.sidebar:
            st.header("ℹ️ About")
            
            if db_status:
                st.success(f"✅ Database: {doc_count} chunks loaded")
            else:
                st.error("❌ Database not found")
            
            st.divider()
            st.markdown("### 👤 Author")
            st.markdown("**Ahmed Pasha**")
            st.caption("Built with ❤️ using RAG")
            
            st.divider()
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            1. Click **Start Chatting with AI**
            2. Ask any question about Insurellm
            3. Get AI-powered answers!
            """)

if __name__ == "__main__":
    main()
