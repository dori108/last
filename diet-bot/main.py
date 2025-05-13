from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# 모델 및 토크나이저 로드
model_id = "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# 요청 데이터 형식 정의
class Prompt(BaseModel):
    goal: str  # 예: "다이어트", "벌크업", "채식"

@app.post("/diet")
def generate_diet(prompt: Prompt):
    input_text = f"오늘의 {prompt.goal} 식단을 추천해줘."
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    diet_plan = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"diet": diet_plan}
