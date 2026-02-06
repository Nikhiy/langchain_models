from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="you are a helpful {domain} expert"),
    HumanMessage(content="explain in simple terms what is {topic}")
])

prompt = chat_template.invoke({
    "domain": "teacher",
    "topic": "AI"
})

print(prompt)