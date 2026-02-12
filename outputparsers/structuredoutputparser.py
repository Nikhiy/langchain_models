from langchain.output_parsers import StructuredOutputParser,ResponseSchema
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
schema=[
    ResponseSchema(name="fact_1",description="fact 1 about the topic"),
    ResponseSchema(name="fact_2",description="fact 2 about the topic"),
    ResponseSchema(name="fact_3",description="fact 3 about the topic")
]
parser=StructuredOutputParser.from_response_schema(schema)
template=PromptTemplate(
    template="give 3 facts about {topic} \n {format_instructions}",
    input_variables=['topic'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt=template.invoke({"topic":"black hole"})
response=model.ask(prompt=prompt)
print(response)