from langchain.llm import OpenAI
from langchain_core.prompts import PromptTemplate

llm=OpenAI(model_name='gpt-3.5-turbo',temprature=0.9)

prompt1=PromptTemplate(
    template="sggest a catchy block title for the {topic}",
    input_variables=['topic']
)

topic=input("enter a topic")
formatter_prompt=prompt1.format(topic=topic)
blog_title=llm.predict(formatter_prompt)
print(blog_title)