from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor, BM25Retriever, PromptNode, PromptTemplate
from haystack.pipelines import Pipeline
from dotenv import load_dotenv
import os

load_dotenv()


def main(pdf_directory: str):
    text_extractor = PDFToTextConverter(remove_numeric_tables=True)
    preprocessor = PreProcessor(
        split_by='word',
        split_respect_sentence_boundary=True,
        split_length=200,
        split_overlap=20,
    )
    document_store = InMemoryDocumentStore(use_bm25=True)

    injest = Pipeline()
    injest.add_node(text_extractor, name="text_extractor", inputs=["File"])
    injest.add_node(preprocessor, name="preprocessor", inputs=["text_extractor"])
    injest.add_node(document_store, name="document_store", inputs=["preprocessor"])

    path = Path(pdf_directory)
    files = list(path.glob('*.pdf'))
    files_metadata = [{"name": file.name, "path": file} for file in files]
    injest.run(file_paths=files, meta=files_metadata)

    retriever = BM25Retriever(document_store=document_store, top_k=5)
    prompt = PromptTemplate(
        name="question",
        prompt_text="""Answer the question based on the context below.
        Context: {join(documents)}\n\n
        Question: {query}\n\n
        Answer:"""
    )
    prompt_node = PromptNode(model_name_or_path="text-davinci-003",
                             api_key=os.environ["OPENAI_API_KEY"], default_prompt_template=prompt)
    chatbot = Pipeline()
    chatbot.add_node(retriever, name="retriever", inputs=["Query"])
    chatbot.add_node(prompt_node, name="prompt", inputs=["retriever"])

    while True:
        query = input("\n\nEnter a question: ")

        if query == "exit":
            break

        response = chatbot.run(query=query)
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
