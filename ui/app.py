import sys
import os
import json
import streamlit as st

# Ensure root path is accessible when running from ui/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.loader import Loader
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever
from model.agent import SLMAgent

# Set Page Configuration for maximum width and title
st.set_page_config(page_title="SLM Code Assistant", page_icon="code", layout="wide")

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

    /* Custom Feedback Button styles */
    .stButton>button {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stButton>button:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
        color: #58a6ff !important;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to save preference feedback for RLHF/DPO
def save_feedback(prompt: str, chosen: str, rejected: str):
    dir_path = os.path.join(st.session_state.target_repo, "training_data", "preferences")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "preference_data_rlhf.jsonl")
    data = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "source": "streamlit_rlhf_feedback"
        }
    }
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "target_repo" not in st.session_state:
    st.session_state.target_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "model" not in st.session_state:
    st.session_state.model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")

# Render dynamic Sidebar settings first
with st.sidebar:
    st.header("Settings")
    new_repo = st.text_input("Local Repository Path", value=st.session_state.target_repo, help="The absolute path to the Python project you want to analyze.")
    
    if new_repo != st.session_state.target_repo:
        st.session_state.target_repo = new_repo
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

    new_model = st.text_input("Ollama Model Name", value=st.session_state.model, help="The model name served by local Ollama.")
    if new_model != st.session_state.model:
        st.session_state.model = new_model
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

    if not os.path.isdir(st.session_state.target_repo):
        st.warning(f"Directory not found: {st.session_state.target_repo}")
        st.stop()

# Cache the heavy resource loading
@st.cache_resource
def load_backend(repo_path, model_name):
    loader = Loader(repo_path)
    chunks = loader.process_directory()
    repo_map_gen = RepoMapGenerator()
    repo_map_string = repo_map_gen.generate_map(chunks)
    
    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)
    
    # Init Agent
    agent = SLMAgent(repo_map_string=repo_map_string, model=model_name, hallucination_check=True)
    agent.set_retriever(retriever)
    agent.set_chunks(chunks)
    
    return agent, repo_map_string, chunks

st.title("Python SLM Agent Assistant")
st.caption(f"powered by {st.session_state.model} (Local via Ollama) & Hybrid Qdrant Search")

try:
    with st.spinner(f"Analyzing {st.session_state.target_repo} and Booting Agent..."):
        agent, repo_map, chunks = load_backend(st.session_state.target_repo, st.session_state.model)
except Exception as e:
    st.error("### Backend Initialization Failed")
    st.markdown("Could not scan the repository or initialize the SLM backend. See details below:")
    st.exception(e)
    st.stop()

# Render dynamic Sidebar
with st.sidebar:
    st.header("Repository Map")
    st.success(f"{len(chunks)} Logical Chunks Extracted via AST")
    
    with st.expander("View Full Map", expanded=True):
        st.code(repo_map, language="markdown")
        
    st.divider()
    st.info(f"The SLM Agent ({st.session_state.model}) has autonomous access to Semantic Search and Exact-Match Grep. It connects via local Ollama API on port 11434.")

# Initialize Chat Memory
# Render chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display feedback buttons for assistant messages
        if message["role"] == "assistant":
            feedback = message.get("feedback")
            prompt_text = message.get("prompt")
            
            if prompt_text:  # Ensure we have the prompt to pair with
                if not feedback:
                    st.markdown("<div style='border-top: 1px solid rgba(240, 246, 252, 0.1); margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                    col1, col2, _ = st.columns([0.15, 0.22, 0.63])
                    with col1:
                        if st.button("👍 Like", key=f"like_{idx}", help="Like this response"):
                            message["feedback"] = "like"
                            save_feedback(prompt_text, message["content"], "Agent failed to answer correctly or answer was suboptimal.")
                            st.success("Liked!")
                            st.rerun()
                    with col2:
                        if st.button("👎 Correction", key=f"dislike_{idx}", help="Provide a correct/preferred response"):
                            message["feedback"] = "dislike_pending"
                            st.rerun()
                elif feedback == "like":
                    st.markdown("<div style='border-top: 1px solid rgba(240, 246, 252, 0.1); margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                    st.markdown("<small style='color: #58a6ff;'>👍 Liked (Added to preference dataset)</small>", unsafe_allow_html=True)
                elif feedback == "dislike_pending":
                    st.markdown("<div style='border-top: 1px solid rgba(240, 246, 252, 0.1); margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                    st.markdown("<small style='color: #ff7b72;'>👎 Correction required:</small>", unsafe_allow_html=True)
                    with st.form(key=f"form_{idx}"):
                        correction = st.text_area("Correct/Preferred Response:", value="", key=f"corr_text_{idx}", help="How should the agent have answered?")
                        submit = st.form_submit_button("Save Feedback")
                        if submit:
                            if correction.strip():
                                message["feedback"] = "dislike"
                                message["correction"] = correction
                                save_feedback(prompt_text, correction, message["content"])
                                st.success("Correction saved to preference dataset!")
                                st.rerun()
                            else:
                                st.warning("Please enter a correction.")
                elif feedback == "dislike":
                    st.markdown("<div style='border-top: 1px solid rgba(240, 246, 252, 0.1); margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                    st.markdown("<small style='color: #ff7b72;'>👎 Disliked (Correction added to preference dataset)</small>", unsafe_allow_html=True)
                    if message.get("correction"):
                        st.info(f"**Your correction:** {message['correction']}")

# If the last message is from the user, generate the assistant response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Agent is exploring tools and thinking..."):
            try:
                response = agent.ask(user_prompt)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "prompt": user_prompt,
                    "feedback": None,
                    "correction": None
                })
                st.rerun()
            except Exception as e:
                error_msg = f"**Execution Failed.** Please make sure your local Ollama daemon is running with `ollama run {st.session_state.model}`."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"{error_msg}\n\n*Error Detail: {e}*",
                    "prompt": user_prompt,
                    "feedback": None,
                    "correction": None
                })
                st.rerun()

# User Input Logic
if prompt := st.chat_input("Ask a question about the codebase... (e.g. 'How does AST chunking work?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()
