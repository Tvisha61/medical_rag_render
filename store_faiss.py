# store_faiss.py

import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pickle
import os

# Load PDF
pdf_path = "who_dengue_guidelines.pdf"
doc = fitz.open(pdf_path)

# Extract texts and metadata
texts, pages = [], []
for page_num, page in enumerate(doc, start=1):
    text = page.get_text().strip()
    if text:
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        texts.extend(chunks)
        pages.extend([page_num] * len(chunks))

# Convert to LangChain documents
documents = [Document(page_content=t, metadata={"page": p}) for t, p in zip(texts, pages)]

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
db = FAISS.from_documents(documents, embedding_model)

# Save it
db.save_local("faiss_index")

# Optional: Save meta
with open("faiss_metadata.pkl", "wb") as f:
    pickle.dump({"texts": texts, "pages": pages}, f)

print("✅ FAISS index created and saved.")
