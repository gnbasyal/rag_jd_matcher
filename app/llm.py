from app.config import settings


def build_llm(api_key: str, provider: str = "openai"):
    """Build an LLM instance using the caller-supplied key and provider."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=api_key,
            # temperature=0,
            top_k=1,
            top_p=0,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=api_key,
        temperature=0,
        seed=42,
        top_p=0,
    )
