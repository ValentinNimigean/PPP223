import sys
import os
import streamlit as st

# Ensure root path is accessible when running from ui/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.loader import Loader
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever
from model.agent import SLMAgent

# Set Page Configuration for maximum width and title
st.set_page_config(page_title="SLM Code Assistant", page_icon="💻", layout="wide")

# Inject Custom Framework CSS for "Glassmorphic" Premium UI
st.markdown("""
<style>
    /* Dark Premium Base */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Subtle Sidebar styling */
    .stSidebar {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Chat bubbles: Assistant */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #21262d;
        border: 1px solid rgba(240, 246, 252, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    
    /* Chat bubbles: User */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: rgba(31, 111, 235, 0.1);
        border: 1px solid rgba(31, 111, 235, 0.3);
    }

    h1, h2, h3 {
        color: #58a6ff;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache the heavy resource loading so it persists across UI re-renders
@st.cache_resource
def load_backend():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    loader = Loader(root_dir)
    chunks = loader.process_directory()
    repo_map_gen = RepoMapGenerator()
    repo_map_string = repo_map_gen.generate_map(chunks)
    
    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)
    
    # Init Agent
    agent = SLMAgent(repo_map_string=repo_map_string)
    agent.set_retriever(retriever)
    
    return agent, repo_map_string, chunks

st.title("🚀 Python SLM Agent Assistant")
st.caption("powered by Qwen-2.5-Coder (Local) & Hybrid Qdrant Search | Built by Team Mobtrap")

with st.spinner("Analyzing Repository and Booting Agent..."):
    agent, repo_map, chunks = load_backend()

# Render dynamic Sidebar
with st.sidebar:
    st.header("📂 Repository Map")
    st.success(f"{len(chunks)} Logical Chunks Extracted via AST")
    
    with st.expander("View Full Map", expanded=True):
        st.code(repo_map, language="markdown")
        
    st.divider()
    st.info("The SLM Agent has autonomous access to Semantic Search and Exact-Match Grep. It connects via local Ollama API on port 11434.")

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Logic
if prompt := st.chat_input("Ask a question about the codebase... (e.g. 'How does AST chunking work?')"):
    # Append to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Agent response
    with st.chat_message("assistant"):
        with st.spinner("Agent is exploring tools and thinking..."):
            try:
                response = agent.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"⚠️ **Connection Failed.** Please make sure your local Ollama daemon is running with `ollama run qwen2.5-coder:7b`.\n\n*Error Detail: {e}*"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
