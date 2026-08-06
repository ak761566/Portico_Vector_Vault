from typing import Dict, Any, List
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import joinedload
from operator import itemgetter
from ingestion import DocumentProcessor

from llm_factory import LLMFactory
from vector_store import VectorStoreManager

class RAGEngine:
    def __init__(self):
        """
            Orchestrates the entire Conversational RAG execution path.
            Automatically binds the chosen LLM and loads the vector search index.
        """
        # 1. Initialize our components
        self.llm = LLMFactory.get_llm()
        self.vector_manager = VectorStoreManager()
        self.processor = DocumentProcessor()
        self.raw_docs = self.processor.process_directory(Path("./source_docs"))
        self.retriever = self.vector_manager.get_hybrid_retriever(self.raw_docs, vector_k=15, bm25_k=15)
        #self.retriever = self.vector_manager.get_retriever(search_k=20)

        # 2. Build the history-aware query re-writer prompt
        # This transforms conversational follow-ups into sharp standalone questions

        re_write_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is. "
        )

        self.re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', re_write_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ('human', "{question}")
            ]
        )

        # Define the standalone question sub-chain
        self.question_generator = self.re_write_prompt | self.llm | StrOutputParser()

        # 3. Build the core QA system prompt
        # In rag_engine.py: Replace the existing qa_system_prompt assignment

        qa_system_prompt = (
            "You are an enterprise knowledge assistant for the Portico project.\n"
            "Your knowledge base contains two types of documents:\n"
            "1. Business & Client Documentation (.txt/.md files outlining client profiles, business goals, and SLAs).\n"
            "2. Technical Error Spreadsheets (.xlsx files detailing error codes, stream setups, and resolution steps).\n\n"
            "CRITICAL EXECUTION INSTRUCTIONS:\n"
            "- GENERAL BUSINESS QUERIES: If the user asks about the client, business overview, scope, or project goals, "
            "synthesize the answer strictly from the Business & Client Documentation in the Context.\n"
            "- TECHNICAL QUERIES: If the user asks about error codes (e.g., C550), streams, or fixes, summarize the spreadsheet logs.\n"
            "- MULTI-RECORD AGGREGATION: Include details from all matching documents in a clean, human-readable format.\n"
            "- OUT OF SCOPE: If the query is completely unrelated to the project documents (e.g., baking recipes, sports, general world trivia), "
            "respond EXACTLY with: 'This query is outside the scope of the project documentation database.'\n\n"
            "Context:\n{context}"
        )

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ('system', qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ('human', "{question}")

        ])


    def _format_docs(self, docs: List) -> str:
        return "\n\n".join(doc.page_content for doc in docs)


    def get_chain(self):
        def get_standalone_question(input_dict: dict)->str:
            if input_dict.get("chat_history"):
                return self.question_generator.invoke(input_dict)
            return input_dict.get("question","")

        standalone_question_runnable = RunnableLambda(get_standalone_question)

        retrieval_chain = {
            "context" : standalone_question_runnable | self.retriever | self._format_docs,
            "chat_history" : itemgetter("chat_history"),
            "question" : get_standalone_question
        }

        full_rag_chain = retrieval_chain | self.qa_prompt | self.llm | StrOutputParser()
        return full_rag_chain






