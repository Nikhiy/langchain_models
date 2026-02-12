from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = ""
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import stroutputparser
from langchain_core.runnables import RunnableSequence,RunnableParallel


promptu=PromptTemplate(
    prompt="write a tweet about {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    prompt="wite a linked post about {topic}",
    input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

parser=stroutputparser()

parallel_chain=RunnableParallel({
    "tweet":RunnableSequence([promptu,model,parser]),
    "linked_post":RunnableSequence([prompt2,model,parser])
})
#outut will be a dict with keys "tweet" and "linked_post"
result=parallel_chain.invoke({'topic':'ai'})
print(result)