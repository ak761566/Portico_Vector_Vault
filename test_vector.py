from pathlib import Path
from ingestion import DocumentProcessor
from vector_store import VectorStoreManager
from rag_engine import RAGEngine

engine = RAGEngine(default_provider="bedrock")

test_query = "List all workbench errors recorded by the developer Dinesh Singh across all streams"

retrieved_docs = engine.retriever.invoke(test_query)
print(f"Total chunks retrieved by the retriever {len(retrieved_docs)}")
dinesh_in_context = [doc for doc in retrieved_docs if "dinesh" in doc.page_content.lower()]
print(f"Total chunks contain dinesh {len(dinesh_in_context)}")
for doc in dinesh_in_context:
    print(f"Row {doc.metadata.get("row")}")
#print(retrieved_docs)

# 1. Inspect what the retriever actually pulled
for idx, doc in enumerate(retrieved_docs, start=1):
    owner = doc.metadata.get("error_code", "N/A")
    sheet = doc.metadata.get("sheet", "N/A")
    #print(f"[{idx}] Sheet {sheet} | Code {owner} | content_snippet {doc.page_content[0:60]}...")


# 2. Inspect formatted context
formatted_context = engine._format_docs(retrieved_docs)
#print(f"Formated context size {len(formatted_context)} character.")

# 3. Test generation directly
#print("\n Invoking LLM")
response = engine.chain.invoke({"question":test_query, "chat_history":[]})
#print("\n" + "=" * 60)
#print(response)
#print("=" * 60)