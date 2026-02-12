from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = ""
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import stroutputparser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

prompt=PromptTemplate(
    prompt="write a joke about {topic}",
    input_variables=["topic"]
)

llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)
parser=stroutputparser()

chain1=RunnableSequence([prompt,llm,parser])

promt1=PromptTemplate(
    prompt="explain the joke: {joke}",
    input_variables=["joke"]
)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explain':RunnableSequence([promt1,llm,parser])
})

final_chain=RunnableSequence([chain1,parallel_chain])

ans=final_chain.invoke({'topic':'AI'})
print(ans)