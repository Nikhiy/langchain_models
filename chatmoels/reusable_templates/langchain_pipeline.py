from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


class LocalLangChainLLM:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,   # change to float16 if GPU
            device_map="auto"
        )

        # create HF pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

        # convert to LangChain LLM
        self.llm = HuggingFacePipeline(pipeline=pipe)

    # reusable function
    def ask(self, prompt: str):
        return self.llm.invoke(prompt)


# 🔹 create model object
model = LocalLangChainLLM()

# 🔹 use it
print(model.ask("Explain artificial intelligence in simple terms"))
