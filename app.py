from rag_utils import ask_question

print("📘 Medical RAG Chatbot (LangChain + FlanT5). Type 'exit' to quit.")

while True:
    query = input("\n❓ Ask a question: ").strip()
    if query.lower() == "exit":
        break

    pages, preview, answer = ask_question(query)
    if not pages:
        print("\n⚠️ I don't know from the PDF.")
    else:
        print(f"\n📄 Pages: {pages}")
        print(f"📚 Context Preview: {preview}...")
        print(f"\n✅ Answer: {answer}")
