from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .config import config
from .ingest_data import create_retriever_from_pdf

system_prompt = "You are a helpful assitant responsible for answering user queries based on the context provided."
user_prompt = """Use the following pieces of information to answer the user's question.
Answers should be short and to the point.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

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
    return chain


def main(pdf_path):
    chain = create_qa_chain(pdf_path)

    print("`q` to quit")
    while True:
        question = input("Ask a question: ")
        if question == "q":
            break
        answer = chain.invoke(question)
        print(answer)


if __name__ == "__main__":
    from argparse import ArgumentParser

    argument_parser = ArgumentParser()
    argument_parser.add_argument("-p", "--pdf", type=str, help="PDF path")
    args = argument_parser.parse_args()

    if not args.pdf:
        print("No PDF path provided, using sample PDF")
        args.pdf = "data/handbook.pdf"

    main(args.pdf)
