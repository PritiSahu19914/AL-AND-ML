from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pymongo import MongoClient
import os

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCU_d3JGmyXYj6h1Wd2k-iw5F_qtIiacbY"

# Flask setup
# app = Flask(__name__)
# CORS(app)

# Connect to MongoDB
# client = MongoClient("mongodb://localhost:27017")
# db = client["petshop"]

# Load FAISS vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# LangChain QA
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0),
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

while True:
    query=input("user:")
    if query.lower() in ['exit']:
        print('----------')
        break
    response=qa_chain.invoke(query)
    print('ai:',response)

# Detect query type and collection
# def detect_query_type_and_collection(question, collection_names):
#     question_lower = question.lower()
    
#     if "how many" in question_lower or "count" in question_lower:
#         query_type = "count"
#     elif "list" in question_lower or "show all" in question_lower:
#         query_type = "list"
#     else:
#         query_type = "semantic"
    
#     matched_collection = None
#     for name in collection_names:
#         if name.lower() in question_lower:
#             matched_collection = name
#             break

#     return query_type, matched_collection

# # Chat API
# @app.route("/api/chat", methods=["POST"])
# def chat():
#     user_input = request.json.get("question")
#     if not user_input:
#         return jsonify({"error": "No question provided"}), 400

#     try:
#         collection_names = db.list_collection_names()
#         query_type, matched_collection = detect_query_type_and_collection(user_input, collection_names)

#         if query_type == "count" and matched_collection:
#             count = db[matched_collection].count_documents({})
#             return jsonify({"answer": f"There are {count} documents in the '{matched_collection}' collection."})

#         elif query_type == "list" and matched_collection:
#             docs = db[matched_collection].find().limit(5)
#             result = []
#             for doc in docs:
#                 doc.pop("_id", None)
#                 result.append(doc)
#             return jsonify({"answer": result})

#         answer = qa_chain.run(user_input)
#         return jsonify({"answer": answer})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# Run server
# if __name__ == "__main__":
#     app.run(port=5001)
