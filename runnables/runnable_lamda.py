from langchain.schema.runnable import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = ""
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import stroutputparser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
def word_count(text: str) -> int:
    return len(text.split())

runnable_word_count = RunnableLambda(word_count)

prompt=PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"]
)
llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

parser=stroutputparser()

chain=RunnableSequence([prompt,llm,parser])

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_cnt':RunnableLambda(word_count)
})

final_chain=RunnableSequence([chain,parallel_chain])
result=final_chain.invoke({'topic':'ai'})
print(result)