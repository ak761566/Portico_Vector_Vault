from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


template = ChatPromptTemplate.from_messages(
    [
        ("system", "You must answer like a 17th-century pirate."),
        ("human","where is the {location}?")
    ]
)

# Format the template dynamically with real data
compiled_messages1 = template.format_messages(location="Caribbean Sea")
compiled_messages2 = template.format_messages(location="London")

print("---- Compiled structure sent to LLM ----")
print(compiled_messages1)
for msg in compiled_messages1:
    print(f"Role : {msg.type.upper()} | Content: {msg.content}")

print(compiled_messages2)
for msg in compiled_messages2:
    print(f"Role : {msg.type.upper()} | Content: {msg.content}")

