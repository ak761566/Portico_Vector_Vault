from rag_engine import RAGEngine
from langchain_core.messages import HumanMessage, AIMessage


if __name__ == "__main__":
    print("Initialize full RAG Core...")
    engine = RAGEngine()
    #engine.configure_llm(provider="groq")
    chain = engine.get_chain()

    # Mock conversation memory state
    # Let's pretend the user already asked about a task, and the system responded

    mock_history = [
        HumanMessage(content="List all workbench errors recorded by the developer 'Dinesh Singh' across all streams."),
        AIMessage(content="It is related to content archival.")
    ]

    # Follow-up question referencing the past conversation turn ("its status")
    follow_up=f"And summarize the required fix for each."

    print(f"\n Test input: {follow_up}")
    print("processing and retrieving context....")

    response  = chain.invoke(
        {
            "chat_history": mock_history,
            "question" : follow_up
        }
    )

    print(f"\n AI Engine response: {response}")