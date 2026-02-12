from langchain.schema.runnable import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = ""
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import stroutputparser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch

prompt1=PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="summarize the following {topic}",
    input_variables=["topic"]
)

model=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)
parser=stroutputparser()

report_gen_chain=RunnableSequence([prompt1,model,parser])

branch_chain=RunnableBranch(
    (lambda x:len(x.split())>100, RunnableSequence([prompt2,model,parser])),
    RunnablePassthrough()
)

final_chain=RunnableSequence([report_gen_chain,branch_chain])
result=final_chain.invoke({'topic':'nikhil'})
print(result)