import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output


class CustomHybridRetriever(Runnable):
    def __init__(self, faiss_retriever, bm25_retriever, top_k: int=10, rrf_constant: int=60):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.rrf_constant = rrf_constant

    def invoke(self, query:str, config: Optional[RunnableConfig]=None)-> List[Document]:
        # 1. Execute both retrievers concurrently to minimize latency
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_faiss = executor.submit(self.faiss_retriever.invoke, query)
            future_bm25  = executor.submit(self.bm25_retriever.invoke, query)
            faiss_result = future_faiss.result()
            bm25_result = future_bm25.result()

        # 2. Genuine Reciprocal Rank Fusion (RRF)
        # Key: doc.page_content -> Value: accumulated RRF float score

        rrf_scores : defaultdict[str, float] =  defaultdict(float)  #defaultdict(float)  means 0.0
        doc_lookup: dict[str, Document] = {}

        for rank, doc in enumerate(faiss_result, start=1):
            content = doc.page_content
            rrf_scores[content] += 1.0 / (self.rrf_constant) + rank
            doc_lookup[content] = doc

        for rank, doc in enumerate(bm25_result, start=1):
            content = doc.page_content
            rrf_scores[content] += 1.0 / (self.rrf_constant) + rank
            doc_lookup[content] = doc

            # 3. Sort by fused score descending - O(N log N) where N <= 30
        ranked_content = sorted(rrf_scores.keys(), key=lambda c:rrf_scores[c], reverse=True)

        # Slice to top_k to protect LLM context window boundaries
        top_documents = [doc_lookup[c] for c in ranked_content[:self.top_k]]

        # 4. Guardrail: Sanitize alphanumeric tokens
        query_terms  = {word.lower() for word in re.findall(r'\b\w+\b', query)}

        has_relevant_content =   ((doc.page_content.lower()) for doc in top_documents)

        if not top_documents or not has_relevant_content:
            print(f"⚠️ [Guardrail] Blocked query: '{query}' | Tokens: {query_terms}")
            return []

        return top_documents





    # def invoke(self, query: str, config: Optional[RunnableConfig] = None) -> List[Document]:
    #     return custom_ensemble_search(query)