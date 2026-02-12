from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = ""
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import stroutputparser
from langchain_core.runnables import RunnableSequence


#-------- creating joke -----------------
prompt=PromptTemplate(
    prompt="write a job about {topic}",
    input_variables=["topic"]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

parser=stroutputparser()

# Create a RunnableSequence to connect the prompt, LLM, and output parser
chain=RunnableSequence()
result=chain.invoke({'topic': 'AI'})

print(result.content)


#------------ explaining joke---------------

prompt1=PromptTemplate(
    prompt="create a joke about {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    prompt="explain the joke: {joke}",
    input_variables=["joke"]
)

chain=RunnableSequence([prompt1,llm,parser,prompt2,llm,parser])
print(chain.invoke({'topic': 'AI'}))

