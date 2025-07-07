from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

generator = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
llm = HuggingFacePipeline(pipeline=generator)

def ask_question(query, k=3):
    docs = db.similarity_search_with_score(query, k=k)
    filtered = [(doc, score) for doc, score in docs if score < 1.5]

    if not filtered:
        return None, None, "I don't know from the PDF."

    pages = sorted(set(doc.metadata["page"] for doc, _ in filtered))
    context = "\n\n".join(doc.page_content for doc, _ in filtered)

    prompt = f"""
You are a helpful medical assistant.
ONLY answer using the context below from pages: {pages}.
If the answer is not in the context, say: "I don't know from the PDF."

Context:
{context}

Question:
{query}
"""
    result = llm.invoke(prompt)
    return pages, context[:300], result
