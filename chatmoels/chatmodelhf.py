import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

response = client.text_generation(
    "What is the capital of France?",
    model="google/flan-t5-base",
    max_new_tokens=50
)

print(response)
