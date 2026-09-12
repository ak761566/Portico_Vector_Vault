import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from langchain_community.llms.ollama import Ollama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import api_key, max_retries

env_path = Path(".")/".env"
load_dotenv(dotenv_path=env_path)

class LLMFactory:
    @staticmethod
    def get_llm(provider: str = 'groq', model_name: Optional[str]=None, temperature: float =0.0) -> BaseChatModel:
        """
            Dynamically provisions an LLM client based on environment configurations.
            Returns a unified BaseChatModel object to ensure strict interface compatibility.
        """
        #provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
        #"llama-3.1-8b-instant
        #llama-3.3-70b-versatile
        if provider.lower().strip() == 'groq':
            api_key = os.getenv("GROQ_API_KEY")
            # Pythonic secret resolution order: Environment Var -> Streamlit Secret -> Fail

            if not api_key:
                try:
                    import streamlit as st
                    api_key = st.secrets.get("GROQ_API_KEY")
                except ImportError:
                    pass

            if not api_key:
                ## New model " model="openai/gpt-oss-20b", " if the  groq deprecate  model=llama-3.1-8b-instant
                raise ValueError("Missing 'GROQ API KEY'. Please define it in your environment or .env file.")
            #selected_model = model_name or "llama-3.1-8b-instant"
            selected_model = model_name or "openai/gpt-oss-20b"
            print(f"LLM Factory: Initialize Cloud LLM via GROQ API [{selected_model}]...")

            return ChatGroq(
                groq_api_key=api_key,
                model_name=selected_model,
                temperature=temperature
            )
        # -------------------------------------------------------------
        # 2. Anthropic Provider (Claude Engine)
        # -------------------------------------------------------------
        elif provider.lower().strip() in {"anthropic", "claude"}:
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:
                raise ImportError("langchain-anthropic' is missing.")

            api_key = os.getenv("ANTHROPIC_API_KEY")

            if not api_key:
                try:
                    import streamlit as st
                    api_key = st.secrets.get("ANTHROPIC_API_KEY")
                except ImportError as exc:
                    pass

            if not api_key:
                raise ValueError("Missing 'ANTHROPIC_API_KEY'. Please define it in .env or secrets.")

            selected_model = model_name or "claude-3-5-sonnet-20241022"
            print(f"LLM Factory: Initialize Anthropic Claude engine [{selected_model}]...")

            return ChatAnthropic(anthropic_api_key=api_key, model_name=selected_model, temperature=temperature,
                                 max_tokens=2048, timeout=30.0, max_retries=2)

        elif provider == "local":
            print("LLM Factory: Initializing local gemma3 via ollama....")
            return ChatOllama(
                model="gemma3:4b",
                temperature=0.1
            )
        elif provider == "openai":
            print("LLM Factory: Initializing cloud openAI client...")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key.startswith("your_actual_api_key"):
                raise ValueError("Production Error: Valid open ai key must be provided in the .env file.")

            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                open_ai_key=api_key
            )
        else:
            raise ValueError("Production Error: Unknown LLM_Provider configuration: {provider}")

    
