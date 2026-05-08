from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import get_embeddings

from app.retrieval.vectorstore import create_vectorstore
from app.retrieval.retriever import get_retriever
from app.llm.claude import get_llm
from app.chains.rag_chain import build_rag_chain


def run():
    # Load
    docs = load_pdf(r"D:\AI-ML\My Projects\GenAi-RAG\data\raw\SaiCharan_Pittala_Salesforce_Admin_Resume.pdf")

    # Split
    chunks = split_documents(docs)

    # Embed
    embeddings = get_embeddings()

    # Store
    vectorstore = create_vectorstore(chunks, embeddings)

    # Retrieve
    retriever = get_retriever(vectorstore)

    # LLM
    llm = get_llm()

    # Chain
    rag = build_rag_chain(llm, retriever)

    # Query loop
    while True:
        query = input("\nAsk something: ")
        result = rag({"query": query})

        print("\nAnswer:", result["result"])


if __name__ == "__main__":
    run()