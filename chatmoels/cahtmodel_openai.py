from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI(model="gpt-4")
reply=model.invoke("what is capital of india")
print(reply.content)

