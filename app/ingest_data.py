from uuid import uuid4

import faiss
from langchain.retrievers import EnsembleRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader

from .config import config


def pypdf_parser(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        text += "\n" + page.extract_text()
    return text


def create_retriever_from_pdf(pdf_path: str) -> EnsembleRetriever:
    text = pypdf_parser(pdf_path)
    chunks = config.text_splitter.create_documents([text])

    index = faiss.IndexFlatL2(config.embedding_dimension)

    vector_store = FAISS(
        embedding_function=config.embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(chunks, ids=uuids)

    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": config.vector_search_k}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.bm25_search_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )

    return ensemble_retriever


# def create_faiss_index(documnts: Document) -> FAISS:
#     index = faiss.IndexFlatL2(config.embedding_dimension)
#     vector_store = FAISS(
#         embedding_function=config.embedding_model,
#         index=index,
#         docstore=InMemoryDocstore(),
#         index_to_docstore_id={},
#     )

#     vector_store.add_documents(documnts)

#     return vector_store
#     return vector_store
