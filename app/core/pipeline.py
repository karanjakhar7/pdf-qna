from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .config import config
from .ingest_data import create_retriever_from_pdf
from .logger import logger

system_prompt = "You are a helpful assitant responsible for answering user queries based on the context provided."
user_prompt = """Use the following pieces of information to answer the user's question.
Answers should be short and to the point.
If the question is present in the context word for word, provide the answer from word for word.
If you don't know the answer, just say "Data Not Available", don't try to make up an answer.

Context: {context}

Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", user_prompt),
    ]
)


def create_qa_chain(pdf_path):
    ensemble_retriever = create_retriever_from_pdf(pdf_path)
    chain = (
        {"context": ensemble_retriever, "question": RunnablePassthrough()}
        | prompt_template
        | config.llm
        | StrOutputParser()
    )
    logger.info("QA chain created")
    return chain
