from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_id = "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

class Prompt(BaseModel):
    goal: str

@app.post("/diet")
def generate_diet(prompt: Prompt):
    input_text = f"오늘의 {prompt.goal} 식단을 추천해줘."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    diet_plan = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"diet": diet_plan}
