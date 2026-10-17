import streamlit as st
import os

from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from llm_factory import LLMFactory, UserAWSCredentials
from rag_engine import RAGEngine

env_path = Path(".")/".env"
load_dotenv(dotenv_path=env_path)

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="PyRAG-Portico", page_icon="", layout="centered")
#st.title("Ithaka (Portico) Project Assistant")
st.title("Portico Smart AI Assistant")
st.caption("Enterprise RAG Engine for Documentation Triage & Technical Error Resolution")

# ==========================================
# 2. Engine Cache & Lifecycle
# ==========================================

# Cache the RAG engine initialization in memory across app reruns
@st.cache_resource(show_spinner="Initializing RAG Engine & Loading Indexes...")
def load_rag_engine()->RAGEngine:
    """
        Instantiates the RAG engine once and persists it in memory.
        Prevents costly index reloading on every user chat interaction.
    """
    #return RAGEngine()
    return RAGEngine(default_provider="bedrock")

try:
    engine = load_rag_engine()
except Exception as exc:
    st.error(f"Critical initialization error: {exc}")
    st.stop()

# ==========================================
# 3. Sidebar: Model Selection & Credentials
# ==========================================
#user_credentials: Optional[UserAWSCredentials] = None

# st.sidebar.title("Model Selection")
# #model_choice = st.sidebar.radio("Active Inference Engine:",["Groq", "AWS Bedrock (Claude)"])
# model_choice = st.sidebar.radio("Active Inference Engine:",["AWS Bedrock (Claude)"])
#
# if model_choice == "AWS Bedrock (Claude)":
#     with st.sidebar.expander("AWS Bedrock Key empty", expanded=True):
#         user_access_key = st.text_input("AWS Access Key ID", type="password")
#         user_secret_access_key = st.text_input("AWS Secret Access Key ID", type="password")
#         user_region = "ap-south-1"
#         access_key = user_access_key.strip()
#         secret_access_key = user_secret_access_key.strip()
#
#         if access_key and secret_access_key:
#             try:
#                 user_credentials = UserAWSCredentials(access_key_id=access_key, secret_access_key=secret_access_key, region_name=user_region)
#                 if engine.current_provider != "bedrock" or not engine.is_ready:
#                     with st.sidebar.status("Binding bedrock claude...", expanded=False):
#                         engine.configure_llm(provider="bedrock", aws_credentials=user_credentials)
#                 st.sidebar.success("Powered by claude.")
#             except Exception as exc:
#                 st.sidebar.error(f"Failed to bind bedrock {exc}")
#         else:
#             st.sidebar.error("Enter AWS credentials to activate claude.")
#
# elif engine.current_provider != "groq":
#     engine.configure_llm(provider="groq")
#     st.sidebar.info("Powered by Groq LPU")


# ==========================================
# 3. Sidebar: Architecture Telemetry (No Key Prompts!)
# ==========================================
st.sidebar.title("System Control")

provider_selection = st.sidebar.radio(
    "Active Model Provider:",
    #options=["AWS Bedrock (Claude 3.5 Sonnet)", "Groq LPU (Llama 3.1)"],
    options=["AWS Bedrock (Claude 3.5 Sonnet)"],
    index=0
)

target_provider = "bedrock" if "Bedrock" in provider_selection else "groq"
#engine.configure_llm(provider=target_provider)

if engine.current_provider != target_provider:
    try:
        st.sidebar.status("Swapping inference_provider...", expanded=False)
        engine.configure_llm(provider=target_provider)
        st.sidebar.success(f"Swapped to {target_provider.upper()} engine.")
    except Exception as exc:
        st.sidebar.error(f"Failed to swap provider {exc}")

# if engine.current_provider != target_provider:
#     try:
#         with st.sidebar.status("Swapping inference engine", expanded=False):
#             engine.configure_llm(provider=target_provider)
#         st.sidebar.status(f"Successfully swapped to {target_provider.upper()} engine.")
#     except Exception as exc:
#         st.sidebar.status(f"Failed to switch provider: {exc}")

# user_access_key = os.getenv("AWS_ACCESS_KEY_ID")
# user_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
# user_region = "ap-south-1"
# access_key = user_access_key.strip()
# secret_access_key = user_secret_access_key.strip()

# try:
#     # user_credentials = UserAWSCredentials(access_key_id=user_access_key, secret_access_key=user_secret_access_key, session_token=user_token, region_name=user_region)
#     user_credentials = UserAWSCredentials(access_key_id=access_key, secret_access_key=secret_access_key,
#                                           region_name=user_region)
#     # print(user_credentials)
#     if engine.current_provider != "bedrock" or not engine.is_ready:
#         with st.sidebar.status("Binding bedrock claude...", expanded=False):
#             #engine.configure_llm(provider="bedrock", aws_credentials=user_credentials)
#             engine.configure_llm(provider="bedrock")
#     st.sidebar.success("Powered by claude.")
# except Exception as exc:
#     st.sidebar.error(f"Failed to bind bedrock {exc}")


# ==========================================
# 4. Session State & History Transcript
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 4. Display Existing Chat Transcript
# Streamlit clears the screen on every run; we must manually redraw past messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# ==========================================
# 5. User Query & Streaming Response Loop
# ==========================================
if user_query := st.chat_input("Ask Ithaka (Portico) project related questions."):
    # Render user message on screen instantly
    if not engine.is_ready or engine.chain is None:
        st.warning("Inference engine is not ready. Please check the credentials in sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_query)

    # Render user message on screen instantly
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""


        try:
            # We use .stream() instead of .invoke() for an elite UX experience.
            # Tokens stream onto the screen live as they are generated by the model.
            for chunk in engine.chain.stream(
                    {
                        "chat_history": st.session_state.chat_history,
                        "question": user_query
                    }
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            # Remove cursor block once streaming concludes
            response_placeholder.markdown(full_response)

            # Enforce clean memory state: Reject refusals, guardrail triggers, and template commentary
            refusal_signature = (
                "This query is outside the scope",
                "I notice that your message appears",
                "It look like you may have accidentally",
                "Its look like you have pasted"
            )

            is_clean_turn = (
                bool(user_query.strip())
                and bool(full_response.strip())
                and not any(full_response.startswith(sig) for sig in refusal_signature)
            )

            # 6. Commit the current turn to persistent memory state
            # if user_query.strip() and full_response.strip() and not full_response.startswith("This query is outside the scope"):
            #     st.session_state.chat_history.append(HumanMessage(content=user_query.strip()))
            #     st.session_state.chat_history.append(AIMessage(content=full_response.strip()))
            if is_clean_turn:
                st.session_state.chat_history.append(HumanMessage(content=user_query.strip()))
                st.session_state.chat_history.append(AIMessage(content=full_response.strip()))
            else:
                print("[State Guard] suppressed toxic turn from persisting in chat history")
        except Exception as e:
            response_placeholder.empty()
            st.error(f"An execution error occurred: {str(e)}")






