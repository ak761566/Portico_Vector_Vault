from rag_engine import RAGEngine
from langchain_core.messages import HumanMessage, AIMessage


if __name__ == "__main__":
    print("Initialize full RAG Core...")
    engine = RAGEngine()
    chain = engine.get_chain()

    # Mock conversation memory state
    # Let's pretend the user already asked about a task, and the system responded

    mock_history = [
        HumanMessage(content="What are the core business of Ithaka?"),
        AIMessage(content="It is related to content archival.")
    ]

    # Follow-up question referencing the past conversation turn ("its status")
    follow_up=f"How they use Jira?"

    print(f"\n Test input: {follow_up}")
    print("processing and retrieving context....")

    response  = chain.invoke(
        {
            "chat_history": mock_history,
            "question" : follow_up
        }
    )

    print(f"\n AI Engine response: {response}")