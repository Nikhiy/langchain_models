from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLLM:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto"
        )

    def ask(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# 🔹 use it
llm = LocalLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print(llm.ask("What is AI?"))
