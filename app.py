from typing import Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from llm_factory import LLMFactory, UserAWSCredentials
from rag_engine import RAGEngine

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="PyRAG-Portico", page_icon="", layout="centered")
#st.title("Ithaka (Portico) Project Assistant")
st.title("Portico Smart AI Assistant")
st.caption("Enterprise Knowledge Base Engine with Dynamic provider switching.")

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
    return RAGEngine(default_provider="groq")

try:
    engine = load_rag_engine()
except Exception as exc:
    st.error(f"Critical initialization error: {exc}")
    st.stop()

# ==========================================
# 3. Sidebar: Model Selection & Credentials
# ==========================================
#user_credentials: Optional[UserAWSCredentials] = None

st.sidebar.title("Model Selection")
model_choice = st.sidebar.radio("Active Inference Engine:",
                                    ["Groq", "AWS Bedrock (Claude)"])

if model_choice == "AWS Bedrock (Claude)":
    with st.sidebar.expander("AWS Bedrock Key empty", expanded=True):
        user_access_key = st.text_input("AWS Access Key ID", type="password")
        user_secret_access_key = st.text_input("AWS Secret Access Key ID", type="password")
        # user_token = st.text_input("AWS Session Token (Optional)", type="password")
        # user_region = st.selectbox("Region", options=["ap-south-1", "us-east-1"], index=0)
        user_region = "ap-south-1"
        # print(user_access_key)
        # print(user_secret_access_key)
        access_key = user_access_key.strip()
        secret_access_key = user_secret_access_key.strip()

        if access_key and secret_access_key:
            try:
                #user_credentials = UserAWSCredentials(access_key_id=user_access_key, secret_access_key=user_secret_access_key, session_token=user_token, region_name=user_region)
                user_credentials = UserAWSCredentials(access_key_id=access_key, secret_access_key=secret_access_key, region_name=user_region)
                print(user_credentials)
                if engine.current_provider != "bedrock" or not engine.is_ready:
                    with st.sidebar.status("Binding bedrock claude...", expanded=False):
                        engine.configure_llm(provider="bedrock", aws_credentials=user_credentials)
                st.sidebar.success("Powered by claude.")
            except Exception as exc:
                st.sidebar.error(f"Failed to bind bedrock {exc}")
        else:
            st.sidebar.error("Enter AWS credentials to activate claude.")

elif engine.current_provider != "groq":
    engine.configure_llm(provider="groq")
    st.sidebar.info("Powered by Groq LPU")




    # if "Claude" in model_choice or "Bedrock" in model_choice:
    #     with st.sidebar.expander("AWS Bedrock Key empty", expanded=True):
    #         user_access_key = st.text_input("AWS Access Key ID", type="password")
    #         user_secret_access_key = st.text_input("AWS Secret Access Key ID", type="password")
    #         user_token = st.text_input("AWS Session Token (Optional)", type="password")
    #         user_region = st.selectbox("Region", options=["ap-south-1", "us-east-1"], index=0)
    #
    #         if user_access_key and user_secret_access_key:
    #             user_credentials = UserAWSCredentials(access_key_id=user_access_key, secret_access_key=user_secret_access_key,
    #                                                   session_token=user_token, region_name=user_region)
    #             if not engine.is_ready:
    #                try:
    #                     engine.configure_llm(provider="bedrock", aws_credentials=user_credentials)
    #                     st.sidebar.success("Powered by Claude 3.5 Sonnet")
    #                except Exception as exc:
    #                    st.sidebar.success("Failed to bind bedrock")
    #         else:
    #             st.sidebar.success("Enter AWS credentials to activate.")

    # else:
    #     engine.update_model(provider="groq")
    #     st.sidebar.info("Powered by Groq LPU (Sub-second Latency)")

    #selected_provider = "bedrock" if "Bedrock" in model_choice else "groq"

    # if selected_provider == "bedrock":
    #     if user_credentials:
    #         engine.configure_llm(provider="bedrock", aws_credentials=user_credentials)
    #     else:
    #         engine.chain = None
    # else:
    #     engine.configure_llm(provider="groq")


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

        # Telemetry in Streamlit sidebar or expander : Begin
        # with st.spinner("Analyzing dependencies.."):
        #     # 1. Inspect retrieval directly
        #     retrieved_doc = engine.retriever.invoke(user_query)
        #
        #
        #     with st.expander("Debug Retrival Inspector", expanded=False):
        #         st.write(f"Document Retrieved: {len(retrieved_doc)}")
        #         for idx,doc in enumerate(retrieved_doc):
        #             st.code(f"Chunk {idx + 1}:\n{doc.page_content}")
        #
        #     if not retrieved_doc:
        #         st.warning("Guardrail Alert: No relevant document found..")
        #         st.stop()
        #
        #     try:
        #         response = engine.chain.invoke({"question":user_query, "chat_history":[]})
        #         if not response or not response.strip():
        #             st.error("Model returned empty string. Check LLM provider/token limit")
        #         else:
        #             st.markdown(response)
        #     except Exception as exc:
        #         st.error(f"Execution failed: {type(exc).__name__} - {exc}")
        #         import traceback
        #         st.code(traceback.format_exc())

        # Telemetry in Streamlit sidebar or expander : END

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

            # 6. Commit the current turn to persistent memory state
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=full_response))
        except Exception as e:
            response_placeholder.empty()
            st.error(f"An execution error occurred: {str(e)}")






