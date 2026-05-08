from langchain_groq import ChatGroq

def get_llm():
    api_key = "GROQ_API_KEY"  # Replace with your actual API key or fetch from environment variables

    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found in environment")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.7
    )