from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

prompt=PromptTemplate(
    template="write the summary fo rthe followign poem {poem}",
    input_variables=["poem"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9
)


parser=StrOutputParser()

loader = TextLoader("document_loaders/cricket.txt", encoding="utf-8")
docs = loader.load()

# print(type(docs))

chain=prompt | model | parser

result=chain.invoke({'poem':docs[0].page_content})
print(result)
