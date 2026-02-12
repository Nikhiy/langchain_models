from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.text_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#load the document
loader=TextLoader('state_of_the_union.txt')
documents=loader.load()

#spit the text into smaller chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs=text_splitter.split_documents(documents)

#convert text into embeddings and store in FAISS
vectorstore=FAISS.from_documents(docs,OpenAIEmbeddings())

#create a retriever for fetching the relavent documents
retriever=vectorstore.as_retriever()

#manually retrieve the relevant documents
query="what are the key points in the state document?"
retrieved_documents=retriever.get_relevant_documents(query)

#combine retrieved documents into a single string
combined_docs="\n".join([doc.page_content for doc in retrieved_documents])

#initialize the language model
llm=OpenAI(model_name='gpt-3.5-turbo',temperature=0.9)

#manully pass the retrieved documents to the language model
prompt=f"Based on the following retrieved documents, answer the question: {query}\n\n{combined_docs}"
answer=llm.predict(prompt)

print(answer)

