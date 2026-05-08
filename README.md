# Ask My Docs – RAG-Based AI Knowledge Assistant

An AI-powered document question-answering system built using Retrieval-Augmented Generation (RAG), LangChain, vector databases, and LLM integration.

The application allows users to upload documents, create embeddings, store them in a vector database, and ask contextual questions using semantic search and Large Language Models.

---

## Features

* Document ingestion and preprocessing
* Text chunking and embedding generation
* Semantic search using vector retrieval
* Retrieval-Augmented Generation (RAG)
* Context-aware question answering
* Streamlit-based interactive UI
* Modular AI pipeline architecture
* LLM integration using Groq/Claude APIs

---

## Tech Stack

### AI / LLM

* LangChain
* RAG Architecture
* LLM Integration
* Semantic Search
* Vector Embeddings

### Backend

* Python
* FastAPI / Streamlit

### Vector Database

* FAISS / ChromaDB

### Libraries

* Pandas
* NumPy
* Scikit-learn

---

## Project Architecture

1. Load documents
2. Split text into chunks
3. Generate embeddings
4. Store vectors in vector database
5. Retrieve relevant chunks
6. Send contextual prompt to LLM
7. Generate intelligent response

---

## Folder Structure

```bash
app/
├── chains/
├── ingestion/
├── llm/
├── retrieval/
├── config.py
├── main.py
```

---

## Installation

```bash
git clone https://github.com/your-username/ask-my-docs.git
cd ask-my-docs
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` file:

```env
GROQ_API_KEY=your_api_key
```

Run the application:

```bash
streamlit run app/streamlit_app.py
```

---

## Use Cases

* Enterprise knowledge assistants
* AI document search systems
* Internal chatbot systems
* Intelligent PDF querying
* Organizational learning assistants

---

## Future Improvements

* Multi-document support
* Hybrid search
* Conversation memory
* Cloud deployment
* Authentication system
* Multi-modal document processing

---

## Author

Sai Charan Pittala
AI/ML Engineer | Generative AI | NLP | RAG | Python
