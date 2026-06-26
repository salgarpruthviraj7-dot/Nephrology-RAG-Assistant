from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
def embed(chunks):
    emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vectordatabase = FAISS.from_documents(chunks,emb)
    vectordatabase.save_local("fiass_index_database")
    return "Created a vector database named : faiss_index_database"