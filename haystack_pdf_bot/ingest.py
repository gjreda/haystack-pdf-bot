from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.pipelines import Pipeline
from pathlib import Path



def ingest(pdf_directory: str) -> InMemoryDocumentStore:
    text_extractor = PDFToTextConverter(remove_numeric_tables=True)
    preprocessor = PreProcessor(
        split_by='word',
        split_respect_sentence_boundary=True,
        split_length=200,
        split_overlap=20,
    )
    document_store = InMemoryDocumentStore(use_bm25=True)

    ingest = Pipeline()
    ingest.add_node(text_extractor, name="text_extractor", inputs=["File"])
    ingest.add_node(preprocessor, name="preprocessor", inputs=["text_extractor"])
    ingest.add_node(document_store, name="document_store", inputs=["preprocessor"])

    path = Path(pdf_directory)
    files = list(path.glob("*.pdf"))
    files_metadata = [{"name": file.name, "path": file} for file in files]
    ingest.run(file_paths=files, meta=files_metadata)
    return document_store
