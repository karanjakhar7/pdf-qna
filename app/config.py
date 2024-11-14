import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

load_dotenv(f"{parentdir}/.env")


@dataclass(frozen=True)
class Config:
    chunk_size: int
    chunk_overlap: int
    length_function: callable
    is_separator_regex: bool

    provider: Literal["azure_openai", "openai"] = "azure_openai"
    embedding_dimension: int = 1024
    embedding_model_name: str = "text-embedding-3-small"
    llm_name: str = "gpt-4o-mini"
    temperature: float = 0.0

    vector_search_k: int = 5
    bm25_search_k: int = 5

    # embedding_model: Embeddings = field(init=False, default=None)
    # llm: LLM = field(init=False, default=None)

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            is_separator_regex=self.is_separator_regex,
        )

    @property
    def embedding_model(self) -> Embeddings:
        if self.provider == "azure_openai":
            from langchain_openai import AzureOpenAIEmbeddings

            return AzureOpenAIEmbeddings(
                model=self.embedding_model_name,
                dimensions=self.embedding_dimension,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-07-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=self.embedding_model_name,
                dimensions=self.embedding_dimension,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

    @property
    def llm(self) -> LLM:
        if self.provider == "azure_openai":
            from langchain_openai import AzureChatOpenAI

            return AzureChatOpenAI(
                model=self.llm_name,
                temperature=self.temperature,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-07-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.llm_name,
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
            )


config = Config(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    embedding_dimension=1024,
    provider="azure_openai",
    embedding_model_name="text-embedding-3-small",
    llm_name="gpt-4o-mini",
    temperature=0.0,
    vector_search_k=5,
    bm25_search_k=5,
)
