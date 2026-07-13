from pathlib import Path
from ingestion import DocumentProcessor
from vector_store import VectorStoreManager

if __name__ == "__main__":
    # 1. Process the files
    docs_dir = Path('./source_docs')
    processor = DocumentProcessor()
    chunks = processor.process_directory(docs_dir)

    if chunks:
        # 2. Build the vector database
        v_manager = VectorStoreManager()
        v_manager.build_and_save_index(chunks)

        # 3. Test a mock retrieval query
        retriever = v_manager.get_retriever(search_k=2)
        # Type a query that matches something inside your sample Excel rows!
        test_query = "what is error code C523?"
        result = retriever.invoke(test_query)
        print(f"\n Search Complete. Retrieved {len(test_query)} relevant entries")
        for idx, doc in enumerate(result):
            print(f"\n[Match {idx + 1}] Metadata {doc.metadata}")
            print(f"Content Summary:\n{doc.page_content}")

