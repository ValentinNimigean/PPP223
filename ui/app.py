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
st.set_page_config(page_title="SLM Code Assistant", page_icon="</>", layout="wide")

# Inject Custom Framework CSS for "Glassmorphic" Premium UI
st.markdown("""
<style>
/* === Tokens === */
:root {
  --color-bg: #0f1117;
  --color-surface: #1a1d27;
  --color-border: #2a2d3a;
  --color-accent: #3b82f6;
  --color-text-primary: #f1f5f9;
  --color-text-muted: #94a3b8;
  --color-error: #ef4444;
  --radius-md: 10px;
  --shadow-card: 0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.2);
}

/* === Global === */
.stApp {
    background-color: var(--color-bg);
    color: var(--color-text-primary);
    font-family: 'Inter', -apple-system, sans-serif;
}

/* === Typography === */
h1, h2, h3, h4, h5, h6 {
    color: var(--color-text-primary) !important;
    font-weight: 600 !important;
}
p, span, div, li {
    font-weight: 400;
    color: var(--color-text-primary);
}
.stCaptionContainer * {
    color: var(--color-text-muted) !important;
}
.stMarkdown p {
    color: var(--color-text-primary);
}

/* === Custom Layout Classes === */
.svg-icon {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.svg-icon svg {
    flex-shrink: 0;
}
.page-title {
    margin: 0;
    padding: 0;
    font-size: 2.25rem;
}
.sidebar-title {
    margin: 0;
    font-size: 1.5rem;
}
.error-title {
    font-weight: 600;
}

/* === Sidebar === */
[data-testid="stSidebar"] {
    background-color: var(--color-surface) !important;
    border-right: 1px solid var(--color-border);
}
[data-testid="stSidebar"] > div:first-child {
    background-color: var(--color-surface);
}
[data-testid="stSidebarNav"] {
    display: none;
}

/* === Cards / Containers / Expanders === */
[data-testid="stExpander"] {
    background-color: var(--color-surface);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-card);
    border: 1px solid var(--color-border);
    overflow: hidden;
}
[data-testid="stAlert"] {
    background-color: var(--color-surface) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-card) !important;
    border: 1px solid var(--color-border) !important;
    color: var(--color-text-primary) !important;
}

/* === Buttons === */
.stButton button {
    background-color: var(--color-accent) !important;
    color: var(--color-text-primary) !important;
    border-radius: 6px !important;
    border: none !important;
    transition: filter 150ms ease !important;
}
.stButton button:hover {
    filter: brightness(1.1);
    border: none !important;
}

/* === Inputs and Text Areas === */
.stTextInput input, .stTextArea textarea {
    background-color: var(--color-bg) !important;
    color: var(--color-text-primary) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    box-shadow: 0 0 0 2px var(--color-accent) !important;
    border-color: var(--color-accent) !important;
}

/* === Scrollbars === */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--color-bg);
}
::-webkit-scrollbar-thumb {
    background: var(--color-border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--color-text-muted);
}

/* === Dividers and Spacing === */
hr {
    border-color: var(--color-border) !important;
    margin: 2rem 0 !important;
}

/* === Chat Interface === */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 1.5rem 0 !important;
    margin-bottom: 0 !important;
    display: flex;
    flex-direction: row;
    gap: 1rem;
    border-bottom: 1px solid var(--color-border) !important;
    box-shadow: none !important;
}
[data-testid="stChatMessage-assistant"] {
    justify-content: flex-start;
    line-height: 1.75;
}
[data-testid="stChatMessage-user"] {
    flex-direction: row-reverse !important;
    text-align: right !important;
    border-right: 3px solid var(--color-accent) !important;
    padding-right: 1rem !important;
    margin-right: 0.5rem;
}

/* === Assistant Icon Badge === */
[data-testid="stChatMessage-assistant"] [data-testid="stChatMessageAvatar"] {
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    color: transparent;
}
[data-testid="stChatMessage-assistant"] [data-testid="stChatMessageAvatar"]::after {
    content: '';
    display: block;
    width: 18px;
    height: 18px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="%233b82f6"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}

/* === Chat Send Button SVG Injection === */
[data-testid="stChatInputSubmitButton"] {
    background-color: #0284c7 !important; /* bright blue */
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: auto 4px !important;
    transition: filter 150ms ease !important;
    border: none !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
    filter: brightness(1.1) !important;
    background-color: #0284c7 !important;
}
[data-testid="stChatInputSubmitButton"] svg {
    display: none !important;
}
[data-testid="stChatInputSubmitButton"]::after {
    content: '';
    display: inline-block;
    width: 16px;
    height: 16px;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="%23ffffff"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 12h14M12 5l7 7-7 7"/></svg>');
    background-size: contain;
    background-repeat: no-repeat;
}



/* === Chat Input === */
[data-testid="stChatInput"] {
    background-color: transparent !important;
    border-top: none !important;
    padding: 1rem 2rem !important;
    width: 100%;
    bottom: 0;
}
[data-testid="stChatInput"] > div {
    background-color: #21262d !important;
    border: none !important;
    border-radius: 24px !important;
    padding: 4px 8px 4px 16px !important;
    transition: all 150ms ease;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stChatInput"] > div:focus-within {
    box-shadow: 0 0 0 2px var(--color-accent), 0 4px 16px rgba(0,0,0,0.3) !important;
}
.stChatInput textarea {
    background-color: transparent !important;
    color: var(--color-text-primary) !important;
    border: none !important;
    box-shadow: none !important;
    padding-top: 12px !important;
    padding-bottom: 12px !important;
}
.stChatInput textarea:focus {
    border: none !important;
    box-shadow: none !important;
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

st.markdown("""
<div class="svg-icon">
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"></path>
        <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"></path>
        <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"></path>
        <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"></path>
    </svg>
    <h1 class="page-title">Python SLM Agent Assistant</h1>
</div>
""", unsafe_allow_html=True)

st.caption("powered by Qwen-2.5-Coder (Local) & Hybrid Qdrant Search | Built by Team Mobtrap")

with st.spinner("Analyzing Repository and Booting Agent..."):
    agent, repo_map, chunks = load_backend()

# Render dynamic Sidebar
with st.sidebar:
    st.markdown("""
    <div class="svg-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"></path>
        </svg>
        <h2 class="sidebar-title">Repository Map</h2>
    </div>
    """, unsafe_allow_html=True)
    
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
        st.markdown(message["content"], unsafe_allow_html=True)

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
                
                # Simulate streaming
                import time
                def stream_data(text):
                    for word in text.split(" "):
                        yield word + " "
                        time.sleep(0.04)
                
                st.write_stream(stream_data(response))
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"""
<div class="svg-icon" style="margin-bottom: 1rem;">
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-error)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>
    <span class="error-title">Connection Failed.</span>
</div>

Please make sure your local Ollama daemon is running with `ollama run qwen2.5-coder:7b`.

*Error Detail: {e}*
"""
                st.markdown(error_msg, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
