from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# Set your Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCU_d3JGmyXYj6h1Wd2k-iw5F_qtIiacbY"

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["petshop"]

# Load all collections
collection_names = db.list_collection_names()
docs = []

for name in collection_names:
    collection = db[name]
    for item in collection.find():
        text_parts = []
        
        # Extract string fields only
        for key, value in item.items():
            if isinstance(value, str) and key != "_id":
                text_parts.append(f"{key}: {value}")

        if text_parts:
            combined_text = " | ".join(text_parts)
            docs.append(Document(page_content=combined_text, metadata={"collection": name}))

# Chunk large documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Create embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(split_docs, embedding_model)

# Save vector store
vector_store.save_local("faiss_index")
print("✅ All collections embedded and saved to FAISS.")
