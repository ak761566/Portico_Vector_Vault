from pathlib import Path
from ingestion import DocumentProcessor
from vector_store import VectorStoreManager

if __name__ == "__main__":
    # 1. Process the files
    docs_dir = Path('./source_docs')
    processor = DocumentProcessor()
    chunks = processor.process_directory(docs_dir)
    #print(chunks)
    # Document(metadata={'source': 'source_docs\\1. Core Preservation Pillars (Requi.txt'},
    #          page_content='1. Core Preservation Pillars (Requirements)\nTo maintain certification as a Trusted Digital'
    #                       ' Repository (TDR), the system must satisfy four non-negotiable requirements for every archival '
    #                       'unit:\n•\tUsability: Content must remain readable despite software/hardware '
    #                       'evolution (mitigating file obsolescence).\n•\tAuthenticity: '
    #                       'A permanent, verifiable audit trail must exist for every file from ingestion through storage.'
    #                       '\n•\tDiscoverability: High-precision metadata indexing '
    #                       'using standardized identifiers (DOI, ISSN, ISBN).'),

    if chunks:
        # 2. Build the vector database
        v_manager = VectorStoreManager()
        v_manager.build_and_save_index(chunks)

        # 3. Test a mock retrieval query
        #retriever = v_manager.get_retriever(search_k=2)
        retriever = v_manager.get_hybrid_retriever(chunks)
        # Type a query that matches something inside your sample Excel rows!
        test_query = "what are core preservation pillars?"
        result = retriever.invoke(test_query)
        print(f"\n Search Complete. Retrieved {len(test_query)} relevant entries")
        for idx, doc in enumerate(result):
            print(f"\n[Match {idx + 1}] Metadata {doc.metadata}")
            print(f"Content Summary:\n{doc.page_content}")

