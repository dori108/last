from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch
import json
import os

MODEL_ID = "google/gemma-2b-it"
device = torch.device("cpu")

hf_token = os.getenv("HUGGINGFACE_TOKEN")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=hf_token).to(device)
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)
except Exception as e:
    print(f"[ERROR] Failed to load Gemma model: {e}")
    generator = None

def call_gemma(prompt: str, max_tokens: int = 512) -> str:
    if not generator:
        return "{}"
    try:
        result = generator(prompt, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)[0]["generated_text"]
        return result
    except Exception:
        return "{}"

def extract_json(text: str) -> dict | None:
    stack = []
    start_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    candidate = text[start_idx:i+1]
                    try:
                        return json.loads(candidate)
                    except:
                        continue
    return None
