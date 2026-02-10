# scripts/rag/components/query_transformers.py

import logging
from typing import List, Dict, Any, Optional, Type

# --- Internal RAG system imports ---
try:
    from config import Settings, load_settings
except ImportError as e:
    logger.critical(f"FATAL: Failed to import Settings from config.py in query_transformers module: {e}. "
                    "Ensure config.py is directly importable (e.g., project root in PYTHONPATH). "
                    "LLM-based query transformation will be limited.")
    Settings = None
    load_settings = None

# --- LangChain related imports for LLM ---
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError as e:
    logger.critical(f"FATAL: Missing LangChain components for LLM-based query transformers: {e}. "
                    "Please install `langchain-core`, `langchain-openai`, `langchain-google-genai`.")
    ChatPromptTemplate = None
    Runnable = None
    RunnableLambda = None
    StrOutputParser = None
    ChatOpenAI = None
    ChatGoogleGenerativeAI = None
    BaseChatModel = None

logger = logging.getLogger(__name__)

# --- Component Registry ---
QUERY_TRANSFORMER_REGISTRY: Dict[str, Type['BaseQueryTransformer']] = {}

def register_query_transformer(name: str):
    """Decorator to register query transformer classes dynamically."""
    def decorator(cls: Type['BaseQueryTransformer']):
        if name in QUERY_TRANSFORMER_REGISTRY:
            logger.warning(f"Query transformer '{name}' already registered. Overwriting.")
        QUERY_TRANSFORMER_REGISTRY[name] = cls
        return cls
    return decorator

# --- Base Query Transformer Class ---
class BaseQueryTransformer:
    """
    Abstract Base Class for all query transformation strategies.
    Defines the interface that all concrete query transformer strategies must implement.
    """
    def __init__(self, **kwargs):
        self.config = kwargs
        logger.info(f"Initializing {self.__class__.__name__} with config: {kwargs}")

    def transform_query(self, query: str) -> List[str]:
        """
        Abstract method to transform a single query into one or more queries.
        Args:
            query (str): The original user query.
        Returns:
            List[str]: A list of transformed queries, including the original query.
        """
        raise NotImplementedError

# --- Concrete Query Transformer Strategies ---

@register_query_transformer("multi_query")
class MultiQueryGenerator(BaseQueryTransformer):
    """
    Generates multiple variations of a user's query using an LLM.
    Inspired by: https://python.langchain.com/v0.2/docs/tutorials/rag/#multi-query-retriever
    """
    def __init__(self, llm: Optional[BaseChatModel] = None, num_queries: int = 3, **kwargs):
        super().__init__(**kwargs)

        if ChatPromptTemplate is None or BaseChatModel is None or StrOutputParser is None:
            raise ImportError("LangChain components required for MultiQueryGenerator are not available.")

        self._llm = llm # Allow passing an LLM instance directly
        self.num_queries = num_queries

        if self._llm is None: # If not provided, try to initialize it from settings/config
            logger.warning("No LLM instance provided to MultiQueryGenerator. Attempting to initialize from config.")
            if Settings is None or load_settings is None:
                raise ValueError("Settings required for LLM initialization are not available. Check config.py import.")
            
            app_settings = load_settings()
            global_llm_config = kwargs.get("global_llm_config", rag_config.get("llm", {})) # Pass global rag_config to init

            llm_provider = global_llm_config.get("provider", "openrouter")
            llm_model = global_llm_config.get("model", "gpt-4o-mini")
            llm_temperature = global_llm_config.get("temperature", 0.1) # Lower temp for query generation

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
                    f"LLM for MultiQueryGenerator not properly configured or API keys missing for provider '{llm_provider}'. "
                    "Ensure LLM is passed or correct environment variables are set."
                )
        
        # This prompt is designed to generate similar questions
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 "You are a helpful AI assistant. Your task is to generate {num_queries} additional "
                 "search queries that are similar to the original user question. "
                 "These queries will be used to retrieve more relevant documents. "
                 "Make sure the queries are distinct but capture the same core intent. "
                 "Respond with one query per line, without any numbering or bullet points."
                 "Include the original query as one of the generated queries."),
                ("human", "{question}"),
            ]
        )

        # Build the LangChain chain
        self.query_chain: Runnable = self.prompt | self._llm | StrOutputParser() | RunnableLambda(lambda text: text.split("\n"))

    def transform_query(self, query: str) -> List[str]:
        if self._llm is None:
            raise RuntimeError("MultiQueryGenerator LLM is not initialized.")
        try:
            # Invoke the LLM chain to get a list of generated queries
            generated_queries = self.query_chain.invoke({"question": query, "num_queries": self.num_queries})
            # Ensure the original query is always in the list
            if query not in generated_queries:
                generated_queries.insert(0, query)
            logger.info(f"MultiQueryGenerator transformed '{query}' into: {generated_queries}")
            return generated_queries
        except Exception as e:
            logger.error(f"Error transforming query '{query}' with MultiQueryGenerator: {e}. Returning original query only.", exc_info=True)
            return [query]

# Add other query transformer strategies here if needed, e.g., HyDE, QueryDecomposition