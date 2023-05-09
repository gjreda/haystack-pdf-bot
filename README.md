# haystack pdf chatbot

Prototyping a bot that allows for chat over PDF documents.

Note that this uses BM25 for document retrieval, rather than a semantic approach with embeddings.

## Setup
This uses [Poetry](https://python-poetry.org/) for dependency management. To install dependencies:
```bash
$ poetry install
```

You'll also need to create a `.env` file and add your `OPENAI_API_KEY` to it (see `.env.example`).

## Usage
The command below will run the pipeline on the `papers` directory, which contains a few PDFs. It will then start a REPL where you can ask questions about the PDFs. You can exit the Q&A loop by typing "exit" or cmd/ctrl + c

```python
$ poetry run python haystack_pdf_bot/main.py --pdf_directory=papers
```

## Example
![/static/example.png](/static/example.png)