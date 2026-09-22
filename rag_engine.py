from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import joinedload
from operator import itemgetter

from torch.distributed.elastic.timer import configure

from ingestion import DocumentProcessor

from llm_factory import LLMFactory, UserAWSCredentials
from vector_store import VectorStoreManager

class RAGEngine:
    def __init__(self,default_provider: str= "groq", source_docs: Path = "./source_docs", faiss_index: Path = "./faiss_index"):
        """
            Orchestrates the entire Conversational RAG execution path.
            Automatically binds the chosen LLM and loads the vector search index.
        """
        # 1. Ingest documents and initialize retrieval structures (Heavy RAM operations)
        self.processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager()
        self.raw_docs = self.processor.process_directory(Path("./source_docs"))
        self.retriever = self.vector_manager.get_hybrid_retriever(self.raw_docs, vector_k=15, bm25_k=15)
        #self.retriever = self.vector_manager.get_retriever(search_k=20)

        # 2. Compile static prompts
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

        # 3. Build the core QA system prompt
        # In rag_engine.py: Replace the existing qa_system_prompt assignment

        # qa_system_prompt = (
        #     "You are an enterprise knowledge assistant for the Portico project.\n"
        #     "Your knowledge base contains two types of documents:\n"
        #     "1. Business & Client Documentation (.txt/.md files outlining client profiles, business goals, and SLAs).\n"
        #     "2. Technical Error Spreadsheets (.xlsx files detailing error codes, stream setups, and resolution steps).\n\n"
        #     "CRITICAL EXECUTION INSTRUCTIONS:\n"
        #     "- GENERAL BUSINESS QUERIES: If the user asks about the client, business overview, scope, or project goals, "
        #     "synthesize the answer strictly from the Business & Client Documentation in the Context.\n"
        #     "- TECHNICAL QUERIES: If the user asks about error codes (e.g., C550), streams, or fixes, summarize the spreadsheet logs.\n"
        #     "- MULTI-RECORD AGGREGATION: Include details from all matching documents in a clean, human-readable format.\n"
        #     "- OUT OF SCOPE: If the query is completely unrelated to the project documents (e.g., baking recipes, sports, general world trivia), "
        #     "respond EXACTLY with: 'This query is outside the scope of the project documentation database.'\n\n"
        #     "Context:\n{context}"
        # )

        qa_system_prompt = (
            "You are an enterprise knowledge assistant for the Ithaka (Portico) project.\n"
            "Your knowledge base contains two types of documents:\n"
            "1. Business & Client Documentation (.txt/.md files outlining client profiles, business goals, and SLAs).\n"
            "2. Technical Error Spreadsheets (.xlsx files detailing error codes, stream setups, and resolution steps).\n\n"
            "CRITICAL EXECUTION INSTRUCTIONS:\n"
            "- GENERAL BUSINESS QUERIES: If the user asks about the client, business overview, scope, sla, or project goals, "
            "synthesize the answer strictly from the Business & Client Documentation in the Context.\n"
            "- TECHNICAL QUERIES: If the user asks about error codes (e.g., C550), streams, or fixes, summarize the spreadsheet logs.\n"
            "- MULTI-RECORD AGGREGATION: Include details from all matching documents in a clean, human-readable format.\n"
            "- OUT OF SCOPE: If the query is completely unrelated to the project documents (e.g., baking recipes, sports, general world trivia), "
            "respond EXACTLY with: 'This query is outside the scope of the project documentation database.'\n\n"
            "RAG SOURCE CITATION INSTRUCTIONS:\n"
            "- Every document retrieved in the RAG application context contains a 'source_url' (TinyURL) and a 'canonical_url'.\n"
            "- To support active project discussions, you MUST explicitly list these sources at the absolute bottom of your response.\n"
            "- Format each unique source using sequential numbers and markdown links exactly like this example:\n"
            "  **Source 1:** [Architecture Guidelines](source_url) | [Canonical Backup](canonical_url)\n"
            "- Never hallucinate, predict, or truncate a URL. If a referenced document lacks a source URL in the metadata, list the document title without a hyperlink.\n\n"
            "Context:\n{context}"
        )

        # qa_system_prompt = (
        #     "You are an enterprise knowledge assistant for the Portico project.\n"
        #     "Your knowledge base contains two types of documents:\n"
        #     "1. Business & Client Documentation (.txt/.md files outlining client profiles, business goals, and SLAs).\n"
        #     "2. Technical Error Spreadsheets (.xlsx files detailing error codes, stream setups, and resolution steps).\n\n"
        #     "CRITICAL EXECUTION INSTRUCTIONS:\n"
        #     "- GENERAL BUSINESS QUERIES: If the user asks about the client, business overview, scope, or project goals, "
        #     "synthesize the answer strictly from the Business & Client Documentation in the Context.\n"
        #     "- TECHNICAL QUERIES: If the user asks about error codes (e.g., C550), streams, or fixes, summarize the spreadsheet logs.\n"
        #     "- SOURCE TRACEABILITY & LINK PRESERVATION:\n"
        #     "  * Whenever your answer draws from a document containing a Confluence link or Markdown URL (e.g., [Title](http...)), "
        #     "you MUST preserve and output that EXACT Markdown link verbatim at the end of the relevant section or as a 'Relevant References' section.\n"
        #     "  * Never strip, alter, or invent URLs. If a markdown link exists in the Context for the retrieved entity, it is a mandatory deliverable.\n"
        #     "- MULTI-RECORD AGGREGATION: Include details from all matching documents in a clean, human-readable format.\n"
        #     "- OUT OF SCOPE: If the query is completely unrelated to the project documents (e.g., baking recipes, sports, general world trivia), "
        #     "respond EXACTLY with: 'This query is outside the scope of the project documentation database.'\n\n"
        #     "Context:\n{context}"
        # )

        # qa_system_prompt = (
        #     "You are an enterprise knowledge assistant for the Portico project.\n"
        #     "Your ONLY source of truth is the verified Context provided below.\n\n"
        #     "CRITICAL INSTRUCTIONS FOR MULTI-RECORD / AGGREGATION QUERIES:\n"
        #     "1. EXHAUSTIVE EXTRACTION: If the user asks for 'all errors', 'list all', or queries a specific developer's "
        #     "records (e.g., 'Dinesh Singh'), you MUST scan EVERY [RECORD ENTRY] in the Context.\n"
        #     "2. NO TRUNCATION: Do NOT stop after the first match. Do NOT provide a single example.\n"
        #     "3. STRUCTURED TABLE OUTPUT: When multiple error records match, format your response as a Markdown Table with columns:\n"
        #     "   | Error Code | Developer | Stream Name | Resolution Steps | Upstream Dependencies |\n"
        #     "4. ACCURACY GUARANTEE: Only include records explicitly present in the Context. If 16 distinct records match, list all 16.\n"
        #     "5. OUT OF SCOPE: If Context is empty or the query is unrelated, output EXACTLY:\n"
        #     "   'This query is outside the scope of the project documentation database.'\n\n"
        #     "Context:\n{context}"
        # )
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ('system', qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ('human', "{question}")

        ])

        # self.llm = LLMFactory.get_llm()
        # self.chain = self.get_chain()
        self.question_generator: Optional[Runnable] = None
        self.llm: Optional[BaseChatModel] = None
        self.chain: Optional[Runnable] = None
        self.current_provider: Optional[UserAWSCredentials] = None

        if default_provider and default_provider.lower() != "bedrock":
            try:
                self.configure_llm(default_provider)
            except Exception as exc:
                print(f"[Rag Engine] Default provider is not initialized {exc}")


    def configure_llm(self, provider: str, model_name: Optional[str] = None, aws_credentials: Optional[UserAWSCredentials] = None) -> None:
        self.current_provider = provider
        self.llm = LLMFactory.get_llm(provider=self.current_provider, model_name=model_name, aws_credentials=aws_credentials)

        # Define the standalone question sub-chain
        self.question_generator = self.re_write_prompt | self.llm | StrOutputParser()

        self.chain = self.get_chain()

    # Alias for configure_llm to maintain backwards compatibility
    update_model=configure_llm

    @property
    def is_ready(self):
        return self.chain is not None

    def _format_docs(self, docs: List) -> str:
        if not docs:
            return "No relevant record found"
        formatted_entries = []
        for idx, doc in enumerate(docs, start=1):
            clean_content = doc.page_content.strip()
            formatted_entries.append(f"--[RECORD-ENTRY--#{idx}]---\n{clean_content}")

        return "\n\n".join(formatted_entries)


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

    def update_model(self, provider: str, model_name: Optional[str] = None, aws_credentials: Optional[UserAWSCredentials] = None) -> None:
        print(f"Swapping model engine to provider: '{provider}', model: '{model_name}'")

        # 1. Swap LLM instance pointer
        self.llm = LLMFactory.get_llm(provider=provider, model_name=model_name, aws_credentials=aws_credentials)

        # 2. Re-wire your chain with the new LLM (Reuses existing self.retriever!)
        self.chain = self.get_chain()




