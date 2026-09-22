import os
from dataclasses import dataclass
from typing import Optional
import boto3
from botocore.config import Config

from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from langchain_community.llms.bedrock import Bedrock
from langchain_community.llms.ollama import Ollama
from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai.providers import bedrock
from requests import session
from torch.autograd.profiler_legacy import profile

env_path = Path(".")/".env"
load_dotenv(dotenv_path=env_path)


@dataclass(frozen=True)
class UserAWSCredentials:
    """Immutable in-memory container for user-supplied AWS credentials."""
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None
    region_name: str = "ap-south-1"

    def validate(self):
        access_key = self.access_key_id.strip()
        secret_key = self.secret_access_key.strip()

        print(f"[Received] Access_key_id: Length {len(self.access_key_id)} Prefix {self.access_key_id[:4]}"
                             f"[Received] Access_key_id: Length {len(self.secret_access_key)}  Prefix {self.secret_access_key[:4]}")

        if not access_key or not secret_key:
            raise ValueError(f"AWS access key or secret key cannot be empty.")

        if access_key.startswith("ASIA") and not(self.session_token and self.session_token.strip()):
            raise ValueError("Temporary credentials detected starts with (ASIA) but AWS session token was not provided.")

class LLMFactory:

    # --- config ---
    REGION = "ap-south-1"
    ACCOUNT_ID = "015615541381"
    TEAM = "enterprisesolution"
    PROFILE_NAME = f"team-{TEAM}-claude-sonnet"
    GUARDRAIL_ID = "63dpeulefsba"
    GUARDRAIL_VERSION = "1"
    GUARDRAIL_ARN = f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:guardrail/{GUARDRAIL_ID}"
    # Keep in sync with your DynamoDB PRICE item (USD per 1K tokens)
    PRICE_IN_PER_1K = 0.0033
    PRICE_OUT_PER_1K = 0.0165
    DEFAULT_MAX_TOKENS = 1024

    BEDROCK_CONFIG: Config = Config(
        connect_timeout=10,
        read_timeout=600,
        max_pool_connections=5,
        retries={
            "mode": "adaptive",
            "total_max_attempts": 10,
        },
    )

    @staticmethod
    def _resolve_profile_arn(bedrock_client, profile_name=PROFILE_NAME):
        """Looks up the application inference profile ARN by name."""
        try:
            paginator = bedrock_client.get_paginator("list_inference_profiles")
            for page in paginator.paginate(typeEquals="APPLICATION"):
                for prof in page["inferenceProfileSummaries"]:
                    if prof["inferenceProfileName"] == profile_name:
                        return prof["inferenceProfileArn"]
        except Exception as e:
            raise ImportError(f"Failed to list profiles: {str(e)}")
        return None

    @classmethod
    def get_llm(cls, provider: str = 'bedrock', model_name: Optional[str]=None, temperature: float =0.0,
                aws_credentials: Optional[UserAWSCredentials] = None, guardrail_id: Optional[str] = "z8r83pgj0a1s",
                guardrail_version: Optional[str]="1") -> BaseChatModel:
        """
            Dynamically provisions an LLM client based on environment configurations.
            Returns a unified BaseChatModel object to ensure strict interface compatibility.
        """
        #provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
        # -------------------------------------------------------------
        # 2. Anthropic Provider (Claude Engine)
        # -------------------------------------------------------------
        if provider.lower().strip() in {"bedrock", "aws", "claud-aws"}:
            try:
                import boto3
                from botocore.config import Config
                from langchain_aws import ChatBedrockConverse
            except ModuleNotFoundError as exc:
                raise ImportError(f"Missing AWS runtime package: {exc.name}") from exc

            if aws_credentials is None:
                raise ValueError("AWS credentials must be explicitly provided by the user at runtime.")


            aws_credentials.validate()

            # bedrock_config = Config(
            #     connect_timeout=10,
            #     read_timeout=600,
            #     retries = {"mode":"adaptive", "total_max_attempt":10}
            # )

            session_kwarg = {
                "aws_access_key_id": aws_credentials.access_key_id,
                "aws_secret_access_key": aws_credentials.secret_access_key,
                "region_name":aws_credentials.region_name
            }

            if aws_credentials.session_token:
                session_kwarg["aws_session_token"] = aws_credentials.session_token

            # Step A: Initialize the session
            boto_session = boto3.Session(**session_kwarg)

            # Step B: Create clients using our BEDROCK_CONFIG
            bedrock_mgmt = boto_session.client("bedrock", config=cls.BEDROCK_CONFIG)
            profile_arn = cls._resolve_profile_arn(bedrock_mgmt)

            if not profile_arn:
                raise ValueError(f"Inference profile {profile_arn} could not be discovered.")

            runtime_client = boto_session.client(
                service_name="bedrock-runtime",
                region_name=aws_credentials.region_name.strip(),
                config=cls.BEDROCK_CONFIG
            )

            guardrail_config = {
                "guardrailIdentifier": cls.GUARDRAIL_ARN,
                "guardrailVersion": cls.GUARDRAIL_VERSION
            }


            return ChatBedrockConverse(
                                       client=runtime_client,
                                       modelId=profile_arn,
                                       temperature=temperature,
                                       max_tokens=1024,
                                       guardrail_config=guardrail_config)

        elif provider.lower().strip() == 'groq':
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

    
