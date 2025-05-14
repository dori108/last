from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# 모델 ID 지정 (2B Instruction-Tuned)
model_id = "google/gemma-2b-it"

# 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 디바이스는 CPU로 고정
device = torch.device("cpu")

# 모델 불러오기 (float32로 설정)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32
).to(device)

# 사용자 입력 형식 정의
class Prompt(BaseModel):
    goal: str  # 예: "다이어트", "벌크업", "채식"

# POST 요청 처리
@app.post("/diet")
def generate_diet(prompt: Prompt):
    input_text = f"오늘의 {prompt.goal} 식단을 추천해줘."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    diet_plan = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"diet": diet_plan}
