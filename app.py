import streamlit as st
from dotenv import load_dotenv
import os
from create_retriever import retrieve

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap # LLM pipeline 

load_dotenv()
api = os.getenv("GROQ_API_KEY")

retriever = retrieve()

prompt = ChatPromptTemplate.from_template("""
You are a medical assistant specializing in kidney health.
Answer the user's question in **simple, human-friendly language**.

You MUST:
- Use the context ONLY to understand the topic.
- **Do NOT mention the words "context", "document", or "retrieved".**
- Do NOT say "Based on the context" or "The context mentions".
- Do NOT reference sources.
- Give a clear, friendly explanation suitable for a patient.
- If the context does not contain enough information, simply say "I don't know."

Context for your understanding:
{context}

Question from the user:
{question}

Your helpful answer:
""")

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    api_key=api
)

rag_chain = (
    RunnableMap({
        "context": (lambda x: x["question"]) | retriever,
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
)


st.set_page_config(page_title="Nephrology RAG Assistant", page_icon="🩺", layout="wide")

st.markdown("""
    <h1 style="text-align:center;">🩺 Nephrology Q&A Assistant</h1>
    <p style="text-align:center;">
        Ask questions related to kidney health, nephrology topics, test reports, symptoms, etc.
    </p>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


question = st.chat_input("Ask your nephrology-related question...")

if question:
    
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)


    with st.chat_message("assistant"):
        with st.spinner("Analyzing medical context..."):
            try:
                result = rag_chain.invoke({"question": question})
                answer = result.content
            except Exception as e:
                answer = f"Error: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

