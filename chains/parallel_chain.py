from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable import RunnableParallel

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

model1=LocalLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model2=LocalLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

promp1=PromptTemplate(
    template="generate short and simple text for the {text}",
    input_variables=['text']
)

promp2=PromptTemplate(
    template="generate 5 short quesition ans answer from the {text}",
    input_variables=['text']
)

promp3=PromptTemplate(
    template="merge the notes {notes} adn quiz {quiz} in to a single document",
    input=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=({
    'notes': promp1 | model1 | parser,
    'quiz': promp2 | model2 | parser
})
merge_chain=promp3 | model1 | parser

chain = parallel_chain | merge_chain

result=chain.invoke({'text':'linear regression is a a straight line model'})

print(result)