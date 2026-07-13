from rag_engine import RAGEngine
from langchain_core.messages import HumanMessage, AIMessage


if __name__ == "__main__":
    print("Initialize full RAG Core...")
    engine = RAGEngine()
    chain = engine.get_chain()

    # Mock conversation memory state
    # Let's pretend the user already asked about a task, and the system responded

    mock_history = [
        HumanMessage(content="What is the error message related to the error code C550?"),
        AIMessage(content="It is related to the transformation file problem.")
    ]

    # Follow-up question referencing the past conversation turn ("its status")
    follow_up=f"What is the error code? generate summary for this error code and include resolution steps associated with this error code"

    print(f"\n Test input: {follow_up}")
    print("processing and retrieving context....")

    response  = chain.invoke(
        {
            "chat_history": mock_history,
            "question" : follow_up
        }
    )

    print(f"\n AI Engine response: {response}")