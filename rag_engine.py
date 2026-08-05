from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import joinedload
from operator import itemgetter
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
        self.retriever = self.vector_manager.get_retriever(search_k=20)

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
        qa_system_prompt = (
            "You are a strict, isolated technical retrieval system operating ON-PREM. "
            "Your ONLY source of truth is the provided Context fragments derived from error logs.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. OUT OF CONTEXT QUERIES: If the user's question is general knowledge, conversational chatter, "
            "or unrelated to workbench error codes/project spreadsheets, output EXACTLY:\n"
            "   'This query is outside the scope of the project error log database.'\n"
            "2. MISSING DATA: If the error code or query keyword IS mentioned in the logs, but specific "
            "fields (e.g., 'Resolution Steps') are empty or omitted, state strictly that the record exists "
            "but resolution in not available.\n"
            "3. ZERO PARAMETRIC KNOWLEDGE: Do NOT use prior training knowledge. Do NOT attempt to answer "
            "general software engineering, math, or administrative questions unless present in the Context.\n"
            "4. NO COMPLIMENTS OR ACKNOWLEDGMENTS: Never say 'Good question', 'I understand', or 'Based on the context'.\n\n"
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
            return input_dict.get("questions","")

        retrieval_chain = {
            "context" : get_standalone_question | self.retriever | self._format_docs,
            "chat_history" : itemgetter("chat_history"),
            "question" : get_standalone_question
        }

        full_rag_chain = retrieval_chain | self.qa_prompt | self.llm | StrOutputParser()
        return full_rag_chain






