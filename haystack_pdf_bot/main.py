from dotenv import load_dotenv

from haystack_pdf_bot.ingest import ingest
from haystack_pdf_bot.bot import ask_question


load_dotenv()


def main(pdf_directory: str):
    document_store = ingest(pdf_directory=pdf_directory)

    while True:
        query = input("\n\nEnter a question: ")

        if query == "exit":
            break

        response = ask_question(query=query, document_store=document_store)
        print(response['results'], end="\n\n")


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument('--pdf_directory', help='Path to directory containing PDFs to ingest')
    args = parser.parse_args()

    path = Path(args.pdf_directory)
    if not path.exists():
        raise ValueError(f"Directory {path} does not exist")

    main(pdf_directory=path.name)
