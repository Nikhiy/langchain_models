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


prompt = PromptTemplate(
    template="""<|user|>
Generate 5 interesting facts about {topic}
<|assistant|>""",
    input_variables=["topic"]
)


local_model=LocalLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

model = RunnableLambda(lambda x: local_model.ask(str(x)))

parser=StrOutputParser()

chain=prompt|model|parser

result=chain.invoke({"topic":"black hole"})
print(result)