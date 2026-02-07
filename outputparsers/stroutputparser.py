from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_core.prompts import PromptTemplate


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
template1=PromptTemplate(
    template="write a detailed review on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="write a 5 line summary on \n {text}",
    input_variables=['text']
)

prompt1=template1.invoke({'topic':'black hole'})
result=model.ask(prompt1)
prompt2=template2.invoke({'text':result})
ans=model.ask(prompt2)
print(ans)
