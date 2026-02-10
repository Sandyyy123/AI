# scripts/rag/components/context_distillers.py

import logging
from typing import List, Dict, Any, Optional, Type

# --- Internal RAG system imports ---
try:
    from config import Settings, load_settings
except ImportError as e:
    logger.critical(f"FATAL: Failed to import Settings from config.py in context_distillers module: {e}. "
                    "Ensure config.py is directly importable (e.g., project root in PYTHONPATH). "
                    "LLM-based context distillation will be limited.")
    Settings = None
    load_settings = None

# --- LangChain related imports for LLM ---
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError as e:
    logger.critical(f"FATAL: Missing LangChain components for LLM-based context distillers: {e}. "
                    "Please install `langchain-core`, `langchain-openai`, `langchain-google-genai`.")
    ChatPromptTemplate = None
    Runnable = None
    StrOutputParser = None
    ChatOpenAI = None
    ChatGoogleGenerativeAI = None
    BaseChatModel = None

logger = logging.getLogger(__name__)

# --- Component Registry ---
CONTEXT_DISTILLER_REGISTRY: Dict[str, Type['BaseContextDistiller']] = {}

def register_context_distiller(name: str):
    """Decorator to register context distiller classes dynamically."""
    def decorator(cls: Type['BaseContextDistiller']):
        if name in CONTEXT_DISTILLER_REGISTRY:
            logger.warning(f"Context distiller '{name}' already registered. Overwriting.")
        CONTEXT_DISTILLER_REGISTRY[name] = cls
        return cls
    return decorator

# --- Base Context Distiller Class ---
class BaseContextDistiller:
    """
    Abstract Base Class for all context distillation strategies.
    """
    def __init__(self, **kwargs):
        self.config = kwargs
        logger.info(f"Initializing {self.__class__.__name__} with config: {kwargs}")

    def distill_context(self, context_for_llm: str, query: str) -> str:
        """
        Abstract method to distill/summarize the raw retrieved context.
        Args:
            context_for_llm (str): The raw context retrieved from the vector store.
            query (str): The original user query.
        Returns:
            str: The distilled/summarized context.
        """
        raise NotImplementedError

# --- Concrete Context Distiller Strategies ---

@register_context_distiller("llm_summarizer")
class LLMContextDistiller(BaseContextDistiller):
    """
    Condenses the retrieved context into a more concise form using an LLM.
    This helps address the 'lost in the middle' phenomenon and manages token limits.
    """
    def __init__(self, llm: Optional[BaseChatModel] = None, summary_type: str = "concise", **kwargs):
        super().__init__(**kwargs)

        if ChatPromptTemplate is None or BaseChatModel is None or StrOutputParser is None:
            raise ImportError("LangChain components required for LLMContextDistiller are not available.")

        self._llm = llm
        self.summary_type = summary_type # e.g., "concise", "bullet_points", "key_facts"

        if self._llm is None: # If not provided, try to initialize it from settings/config
            logger.warning("No LLM instance provided to LLMContextDistiller. Attempting to initialize from config.")
            if Settings is None or load_settings is None:
                raise ValueError("Settings required for LLM initialization are not available. Check config.py import.")
            
            app_settings = load_settings()
            global_llm_config = kwargs.get("global_llm_config", rag_config.get("llm", {})) # Pass global rag_config to init

            # Often, a smaller/faster model is sufficient for summarization to save costs/latency
            llm_provider = global_llm_config.get("provider", "openrouter")
            llm_model = global_llm_config.get("context_distiller_model", "gpt-3.5-turbo") # Specific model for distillation
            llm_temperature = global_llm_config.get("context_distiller_temperature", 0.0) # Low temp for factual summary

            if llm_provider == "openrouter" and app_settings.OPENROUTER_API_KEY and app_settings.OPENROUTER_BASE_URL:
                self._llm = ChatOpenAI(
                    openai_api_key=app_settings.OPENROUTER_API_KEY,
                    openai_api_base=app_settings.OPENROUTER_BASE_URL,
                    model_name=llm_model,
                    temperature=llm_temperature,
                )
            elif llm_provider == "openai" and app_settings.OPENAI_API_KEY:
                self._llm = ChatOpenAI(
                    openai_api_key=app_settings.OPENAI_API_KEY,
                    model_name=llm_model,
                    temperature=llm_temperature,
                )
            elif llm_provider == "gemini" and app_settings.GOOGLE_API_KEY:
                self._llm = ChatGoogleGenerativeAI(
                    google_api_key=app_settings.GOOGLE_API_KEY,
                    model=llm_model,
                    temperature=llm_temperature,
                )
            else:
                raise ValueError(
                    f"LLM for LLMContextDistiller not properly configured or API keys missing for provider '{llm_provider}'. "
                    "Ensure LLM is passed or correct environment variables are set."
                )
        
        # Define the prompt for context distillation
        summarize_prompt_templates = {
            "concise": (
                "You are an expert summarizer. Condense the following context into a concise summary "
                "that directly addresses the user's question. Focus on extracting key facts and details "
                "relevant to the question. Do not add any new information. If the context is very brief, "
                "just return the context as is.\n\n"
                "User Question: {query}\n\n"
                "Context: {context}"
            ),
            "key_facts": (
                "Extract only the most important facts from the following context that are directly "
                "relevant to the user's question, and present them as a bulleted list. "
                "Do not infer or add information. If no facts are relevant, state so.\n\n"
                "User Question: {query}\n\n"
                "Context: {context}"
            )
            # Add more summary types if needed
        }
        prompt_template_str = summarize_prompt_templates.get(self.summary_type, summarize_prompt_templates["concise"])

        self.summary_chain: Runnable = (
            ChatPromptTemplate.from_template(prompt_template_str) | self._llm | StrOutputParser()
        )

    def distill_context(self, context_for_llm: str, query: str) -> str:
        if self._llm is None:
            raise RuntimeError("LLMContextDistiller LLM is not initialized.")
        try:
            logger.info(f"Distilling context ({len(context_for_llm)} chars) using '{self.summary_type}' summary type for query: '{query}'")
            distilled_text = self.summary_chain.invoke({"context": context_for_llm, "query": query})
            logger.debug(f"Distilled context (first 200 chars): {distilled_text[:200]}")
            return distilled_text
        except Exception as e:
            logger.error(f"Error distilling context for query '{query}': {e}. Returning original context.", exc_info=True)
            return context_for_llm # Fallback to original context on error