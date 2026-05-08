pyhfrom langchain.vectorstores import FAISS

def create_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)

def save_vectorstore(vs, path="vectorstore/db"):
    vs.save_local(path)

def load_vectorstore(embeddings, path="vectorstore/db"):
    return FAISS.load_local(path, embeddings)