"""
config.py - Centralized, validated environment configuration.
Loads configuration from process environment or local .env file.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class BedrockSetting:
    """Immutable settings container for AWS Bedrock integration."""
    region: str
    profile_arn: str
    provider: str
    base_model_id:str
    guardrail_id: str
    guardrail_version: str
    access_key_id: Optional[str]
    secret_access_key: Optional[str]
    session_token: Optional[str]

def _load_env_file(dotenv_path: Path = Path(".env"))->None:
    if not dotenv_path.is_file():
        return

    for line in dotenv_path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        #print(line.partition("="))
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))




def get_bedrock_settings() -> BedrockSetting:
    _load_env_file()
    region = os.getenv("AWS_DEFAULT_REGION").strip()
    profile_arn = os.getenv("MODEL_ID").strip()
    base_mode_id = os.getenv("BASE_MODEL_ID").strip()
    provider = os.getenv("PROVIDER").strip()
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID").strip()
    guardrail_version = os.getenv("BEDROCK_GUARDRAIL_VERSION").strip()
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID").strip()
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY").strip()
    session_token = os.getenv("AWS_SESSION_TOKEN").strip()

    return BedrockSetting(region=region,
                          profile_arn=profile_arn,
                          provider=provider,
                          base_model_id=base_mode_id,
                          guardrail_id=guardrail_id,
                          guardrail_version=guardrail_version,
                          access_key_id=access_key_id,
                          secret_access_key=secret_access_key,
                          session_token=session_token)



# bedrock_settings = get_bedrock_setting()
# print(bedrock_settings.profile_arn)