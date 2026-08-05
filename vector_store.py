import os
from pathlib import Path
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.playwright.utils import run_async
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.ensemble import EnsembleRetriever

class VectorStoreManager:
    def __init__(self, index_path: str = 'faiss_index', model_name : str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Manages local vector database lifecycle.
        Using 'all-MiniLM-L6-v2' because it is extremely lightweight (~90MB),
        fast on CPU, and performs well for search.
        """
        self.index_path = index_path
        print("Initializing local huggingface embedding engine....")
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)

    def build_and_save_index(self, docs: List[Document]) -> FAISS:
        if not docs:
            raise ValueError("No document provided to index.")
        print(f"Vectorizing  {len(docs)} data chunks. Please wait..")
        vector_db = FAISS.from_documents(docs, self.embedding)

        # Persist to disk as local flat files
        vector_db.save_local(self.index_path)
        print(f"Vector store successfully saved locally to folder: {self.index_path}")
        return vector_db


    def get_retriever(self, search_k: int = 4):
        """Loads the local database index and configures it as a search retriever."""
        if not Path(self.index_path):
            raise FileNotFoundError(f"Vector index path not found. Run build_and_save_index first.")

        # allow_dangerous_deserialization=True is safe here because we created the pickle files locally ourselves

        vector_db = FAISS.load_local(
            self.index_path,
            self.embedding,
            allow_dangerous_deserialization=True
        )

        return vector_db.as_retriever(search_kwarg={"k":search_k})

    def get_hybrid_retriever(self, all_docs: List[Document], vector_k: int=15, bm25_k: int=15):
        if not Path(self.index_path):
            raise FileNotFoundError(f"Vector index path not found. Run build_and_save_index first.")

        vector_db = FAISS.load_local( self.index_path, self.embedding, allow_dangerous_deserialization=True)
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k":vector_k})
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = bm25_k

        # 2. Define an inner execution function that mimics the LangChain Retriever interface
        def custom_ensemble_search(query: str) -> List[Document]:
            # Run both searches in parallel threads
            faiss_results = faiss_retriever.invoke(query)
            bm25_results = bm25_retriever.invoke(query)

            # Reciprocal Rank Fusion (RRF) algorithm simulation:
            # We merge the lists while strictly preserving order and deduplicating rows
            seen_contents = set()
            combined_docs = []

            for doc in faiss_results + bm25_results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    combined_docs.append(doc)

            # Extract basic query terms (ignoring stop words) to verify minimum keyword presence
            query_terms = [word.lower() for word in query.split() if len(word)>2]

            # Check if at least one meaningful query term or identifier exists in the context
            has_relevant_content = any(
                any(term in doc.page_content.lower() for term in query_terms) for doc in combined_docs
            )

            # Interleave results to balance semantic and keyword matches evenly
            # for doc in sorted(list(set(faiss_results + bm25_results)), key=lambda x: x.page_content):
            #     # Unique identifier based on row text contents
            #     if doc.page_content not in seen_contents:
            #         seen_contents.add(doc.page_content)
            #         combined_docs.append(doc)

            if not has_relevant_content:
                return []

            return combined_docs

            # 3. Create a clean wrapper class so it behaves exactly like a LangChain runnable component

        class CustomHybridRetriever:
            def invoke(self, query: str) -> List[Document]:
                return custom_ensemble_search(query)

        return CustomHybridRetriever()
