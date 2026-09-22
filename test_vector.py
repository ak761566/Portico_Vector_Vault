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

    dinesh_docs = [docs for docs in chunks if "dinesh" in docs.page_content.lower()]
    print(f"[*] Length of dinesh docs {len(dinesh_docs)}")

    if not dinesh_docs:
        print("Critical: Dinesh Singh does not exist in source docs")
    else:
        print(f"[*] Sample match {dinesh_docs[0].page_content[:200]}")

    if chunks:
        # 2. Build the vector database
        v_manager = VectorStoreManager()
        v_manager.build_and_save_index(chunks)

        # 3. Test a mock retrieval query
        #retriever = v_manager.get_retriever(search_k=2)
        retriever = v_manager.get_hybrid_retriever(chunks, vector_k=15, bm25_k=15)
        # Type a query that matches something inside your sample Excel rows!
        test_query = "List all workbench errors recorded by the developer Dinesh Singh across all streams"
        result = retriever.invoke(test_query)
        print(f"\n Search Complete. Retrieved {len(result)} relevant entries")
        for idx, doc in enumerate(result):
            matches_name = "dinesh" in doc.page_content.lower()
            print(f"\n[Chunk {idx + 1}] [Contain dinesh: {matches_name}]")
            #print(f"Content Summary:\n{doc.page_content}")

