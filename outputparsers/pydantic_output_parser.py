from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field

class LocalLLM:
    def __init__(self,model_id):
        self.tokenizer=AutoTokenizer.from_pretrained(model_id)
        self.model=AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto"
        )

    def ask(self,prompt):
        inputs=self.tokenizer(prompt,return_tensors="pt")
        outputs=self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.4
        )
        return self.tokenizer.decode(outputs[0],skip_special_tokens=True)
    
model=LocalLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

class Person(BaseModel):
    name:str=Field(description="this is the name of the person")
    age:int=Field(description="this is the age of the person")
    city:str=Field(description="this is the city the person belong to")

parser=PydanticOutputParser(pydantic_object=Person)
template=PromptTemplate(
    template="generate name,age,city of the fictional {place} person \n {format_instructions}",
    input_variables=['place'],
    partial_variabls={'format_instructions':parser.get_format_instructions()}
)

#we first send this promt(using template) to model then we send this to parser to get that way output

chain=template|model|parser

result=chain.invoke({'place':'india'})
print(result)

