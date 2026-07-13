from pathlib import Path

from ingestion import DocumentProcessor

if __name__ == "__main__":
    processor = DocumentProcessor()
    doc_dir = Path("source_docs/Ithaka-Portico-KnowledgeBase.xlsx")

    chunks  = processor.process_directory(doc_dir)

    if chunks:
        print(f"Total chunks generated {len(chunks)}")
        print(f"Sample chunk 1 content : \n {chunks[0]}")
