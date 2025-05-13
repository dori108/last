from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from multiprocessing import Process, Queue
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
    print("[INFO] Gemma model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load Gemma model: {e}")
    generator = None


def call_gemma(prompt: str, max_tokens: int = 512) -> str:
    if not generator:
        return "[ERROR] Gemma model is not loaded."
    try:
        result = generator(prompt, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)[0]["generated_text"]
        return result
    except Exception as e:
        return f"[ERROR] Gemma generation failed: {str(e)}"


def call_gemma_with_timeout(prompt: str, timeout: int = 30) -> str:
    """Gemma 호출을 제한 시간 내에 실행"""
    def worker(q):
        try:
            result = call_gemma(prompt)
            q.put(result)
        except Exception as e:
            q.put(f"[ERROR] Gemma failed: {str(e)}")

    q = Queue()
    p = Process(target=worker, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        return "[TIMEOUT] Gemma took too long and was terminated."

    return q.get()


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
