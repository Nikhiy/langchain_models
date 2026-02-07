from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence


# ---------------------------
# Local LLM wrapper
# ---------------------------
class LocalLLM:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto"
        )

    def generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.4
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# load model
llm = LocalLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


# ---------------------------
# Prompt templates
# ---------------------------
review_prompt = PromptTemplate(
    template="Write a detailed review on {topic}",
    input_variables=["topic"]
)

summary_prompt = PromptTemplate(
    template="Write a 5 line summary on:\n{text}",
    input_variables=["text"]
)


# ---------------------------
# Convert model into runnable
# ---------------------------
model_runnable = RunnableLambda(lambda prompt: llm.generate(prompt))


# ---------------------------
# Chain
# ---------------------------
chain = (
    review_prompt
    | model_runnable
    | summary_prompt
    | model_runnable
)


# ---------------------------
# Run chain
# ---------------------------
result = chain.invoke({"topic": "black hole"})
print(result)
