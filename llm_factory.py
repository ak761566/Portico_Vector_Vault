import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

env_path = Path(".")/".env"
load_dotenv(dotenv_path=env_path)

class LLMFactory:
    @staticmethod
    def get_llm() -> BaseChatModel:
        """
            Dynamically provisions an LLM client based on environment configurations.
            Returns a unified BaseChatModel object to ensure strict interface compatibility.
        """
        provider = os.getenv("LLM_PROVIDER", "local").strip().lower()

        if provider == "local":
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

    
