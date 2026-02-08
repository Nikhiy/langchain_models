from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

class LocalLLM:
    def __init__(self,model_id):
        self.tokenizer=AutoTokenizer.from_pretrained(model_id)
        self.model=AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto"
        )

    def ask(self, prompt):
        if not isinstance(prompt, str):
            prompt = prompt.to_string()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.4
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

model=LocalLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompt1=PromptTemplate(
    template="generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="generate a 5line summary on {text}",
    input_variables=['text']
)

parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({"topic":"unemployment in india"})

print(result)