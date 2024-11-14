from core.pipeline import create_qa_chain


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
