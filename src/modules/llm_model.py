from dotenv import load_dotenv
import os
import logging
from langchain_anthropic import ChatAnthropic
load_dotenv()
logger = logging.getLogger(__name__)

def create_anthropic_llm(
    temperature: float = float(os.getenv("TEMPERATURE", 0.7)),
    max_tokens: int = int(os.getenv("MAX_NEW_TOKENS", 1024)),
 ):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    model = os.getenv("ANTHROPIC_MODEL_ID", "claude-3-haiku-20240307")

    llm = ChatAnthropic(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.info(f"Created Anthropic LLM model: {model}")
    return llm