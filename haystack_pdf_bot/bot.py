from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, PromptNode, PromptTemplate
from haystack.pipelines import Pipeline
from dotenv import load_dotenv
import os

load_dotenv()


def ask_question(query: str, document_store: InMemoryDocumentStore):
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
    return chatbot.run(query=query)
