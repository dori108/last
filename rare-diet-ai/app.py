from flask import Flask, request, jsonify
from gemma_util import call_gemma_with_timeout, extract_json
import os
import json
from huggingface_hub import login
from pubmed_fetcher import process_disease
from flask_cors import CORS

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(hf_token)

app = Flask(__name__)
CORS(app)

def generate_prompt(user_info, meal_type, disease_info, consumed_so_far):
    disease_texts = []
    remaining_nutrients = {"protein": 0, "fat": 0, "carbohydrates": 0, "sodium": 0}

    for d in user_info["disease"]:
        d_data = disease_info.get(d.lower())
        if not d_data:
            continue
        disease_texts.append(f"* {d_data['note']}\n- Avoid: {', '.join(d_data['avoid'])}\n- Safe: {', '.join(d_data['safe'])}")
        limit = d_data.get("nutrition_limit", {})
        for k in remaining_nutrients:
            if k in limit:
                remaining_nutrients[k] += limit[k]

    for k in remaining_nutrients:
        consumed = consumed_so_far.get(k, 0)
        remaining_nutrients[k] = max(remaining_nutrients[k] - consumed, 0)

    prompt = f"""
You are a professional nutritionist and your task is to recommend a {meal_type.upper()} meal for a specific user.
You will be provided with health details, dietary restrictions, and ingredient availability.
Your response must ONLY be a well-formatted JSON object following the exact structure shown below.

User Information:
- Age: {user_info['age']}
- Gender: {user_info['gender']}
- Height: {user_info['height']}cm
- Weight: {user_info['weight']}kg
- Ingredients available: {', '.join(user_info['ingredients'])}

Health notes:
{chr(10).join(disease_texts)}

Remaining daily intake allowance:
- Protein: {remaining_nutrients['protein']}g
- Fat: {remaining_nutrients['fat']}g
- Carbohydrates: {remaining_nutrients['carbohydrates']}g
- Sodium: {remaining_nutrients['sodium']}mg

RESPONSE INSTRUCTIONS — PLEASE FOLLOW STRICTLY:
- You MUST respond with ONLY valid **JSON**.
- DO NOT add any extra text, explanation, comment, greeting, or closing.
- DO NOT omit or rename any keys.
- Every field shown below must be included — even if the value is 0 or an empty string ("").
- Do not use ellipses (...), placeholders, or incomplete values.
- If a value is unknown, write 0 (for numbers) or "" (for strings) — do not leave out any field.

 REQUIRED JSON FORMAT (copy exactly):
{{
  "meal": {{
    "dish": "Grilled Chicken Salad",
    "menu": ["Grilled chicken breast", "Mixed greens", "Cherry tomatoes", "Olive oil dressing"],
    "notes": ["Low-carb, high-protein meal suitable for most dietary restrictions."],
    "calories": 350,
    "protein": 32.5,
    "carbs": 15.0,
    "fat": 12.0
    "sodium": 800
  }}
}}

Strictly valid JSON, without any extra text
"""
    return prompt

@app.route("/generate_diet", methods=["POST"])
def generate_diet():
    try:
        data = request.json
        user = data["user_info"]
        diseases = user.get("disease", [])
        meal_type = data.get("meal_type", "breakfast").lower()
        consumed = data.get("consumed_so_far", {})

        # ... disease_info, prompt 생성 생략 ...

        result = call_gemma_with_timeout(prompt, timeout=30)

        if "[TIMEOUT]" in result or "[ERROR]" in result:
            return jsonify({
                "error": "Gemma 모델 응답 실패 또는 시간 초과",
                "fallback": True
            }), 504

        parsed = extract_json(result)
        if not parsed:
            return jsonify({
                "error": "Gemma 응답을 JSON으로 파싱할 수 없음",
                "fallback": True
            }), 500

        return jsonify({
            "diet": parsed,
            "fallback": False
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
