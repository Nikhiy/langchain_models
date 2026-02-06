from langchain_openai import ChatOpenAI
from typing import TypedDict,Annotated

model=ChatOpenAI()

class Review(TypedDict):
    summary:str
    sentiment:str

class Review2(TypedDict):
    summary1:Annotated[str,"a breief of the review2"]
    sentiment2:Annotated[str,"sentiment of "]#like this we add any thisg like theme:Annoted[str,"give theme of text like that]
structured_model=model.with_structured_output(Review)

result=structured_model.invoke("This movie was engaging and well-paced. The story kept my attention, and the main character was easy to connect with. While some parts felt predictable, the visuals and background music made the experience enjoyable. Overall, it's a solid film worth watching once.")

print(result)
