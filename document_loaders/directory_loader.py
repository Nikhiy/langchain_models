from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

loader=DirectoryLoader(
    path="books",
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
model=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9
)

docs=loader.load()
