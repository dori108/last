from flask import Flask, request, jsonify
import os
import json
from huggingface_hub import login
from pubmed_fetcher import process_disease
from gemma_util import call_gemma, extract_json
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
    data = request.json
    user = data["user_info"]
    diseases = user.get("disease", [])
    meal_type = data.get("meal_type", "breakfast").lower()
    consumed = data.get("consumed_so_far", {})

    # ✅ carbs → carbohydrates 자동 변환
    if "carbs" in consumed:
        consumed["carbohydrates"] = consumed["carbs"]

    disease_info = {}
    for d in diseases:
        disease_info[d.lower()] = process_disease(d)

    prompt = generate_prompt(user, meal_type, disease_info, consumed)
    result = call_gemma(prompt)
    parsed = extract_json(result)

    if parsed and "meal" in parsed:
        parsed["meal"]["meal_type"] = meal_type  # ✅ meal_type 추가
        return jsonify({"meal": parsed["meal"]})
    else:
        return jsonify({"error": "Invalid response from Gemma"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=5000)
