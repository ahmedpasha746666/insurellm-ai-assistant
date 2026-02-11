import streamlit as st
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from litellm import completion
import time

# Page config
st.set_page_config(
    page_title="Insurellm AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Configuration
MODEL = "ollama/llama3.2:latest"
DB_NAME = r"./pre_processed1_db"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "docs"

# Initialize OpenAI client for embeddings
@st.cache_resource
def get_openai_client():
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Initialize ChromaDB
@st.cache_resource
def get_collection():
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    return collection

# Pydantic models
class Result(BaseModel):
    page_content: str
    metadata: dict

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most to least relevant, by chunk id number"
    )

# RAG Functions
def fetch_context_unranked(question, collection, openai_client, k=10):
    """Retrieve relevant chunks for a question"""
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
    """Use AI to reorder chunks by relevance"""
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
    """Rewrite the user's question to be more specific"""
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
    """Complete RAG pipeline"""
    # Step 1: Rewrite query
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

# Streamlit UI
def main():
    # Header
    st.title("🤖 Insurellm AI Assistant")
    st.markdown("Ask me anything about Insurellm - our products, employees, contracts, and company information!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check if database exists
        try:
            collection = get_collection()
            doc_count = collection.count()
            st.success(f"✅ Database loaded: {doc_count} chunks")
        except Exception as e:
            st.error(f"❌ Database not found. Please run the notebook first to create chunks.")
            st.stop()
        
        st.divider()
        
        # Configuration options
        retrieval_k = st.slider(
            "Number of chunks to retrieve",
            min_value=5,
            max_value=30,
            value=15,
            help="More chunks = more context but slower"
        )
        
        use_reranking = st.checkbox(
            "Use AI Reranking",
            value=True,
            help="Reorder chunks by relevance (slower but better)"
        )
        
        show_sources = st.checkbox(
            "Show source documents",
            value=True,
            help="Display which documents were used to answer"
        )
        
        st.divider()
        st.markdown("### Example Questions")
        st.markdown("""
        - How many employees work remotely?
        - What products does Insurellm offer?
        - Tell me about Markellm
        - Who is the CEO?
        - What's the company culture like?
        """)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message and show_sources:
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(message["sources"][:3], 1):
                        st.markdown(f"**{i}. {source['file']}**")
                        st.text(source['preview'])
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about Insurellm..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    openai_client = get_openai_client()
                    collection = get_collection()
                    
                    # Show rewritten query
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
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Prepare sources
                    sources = []
                    for chunk in chunks[:3]:
                        sources.append({
                            "file": chunk.metadata.get('source', 'Unknown').split('/')[-1],
                            "preview": chunk.page_content[:200] + "..."
                        })
                    
                    # Show sources
                    if show_sources:
                        with st.expander("📄 View Sources"):
                            st.markdown(f"**Rewritten Query:** _{rewritten_query}_")
                            st.divider()
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. {source['file']}**")
                                st.text(source['preview'])
                                st.divider()
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("Make sure Ollama is running: `ollama serve`")
    
    # Clear chat button
    if st.session_state.messages:
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
