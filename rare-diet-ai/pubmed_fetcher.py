import json
from pathlib import Path

DISEASE_LIMIT_PATH = "data/disease_limit/disease_limit.json"

def normalize_disease_name(name: str) -> str:
    return name.lower().replace("(", "").replace(")", "").strip()

def load_disease_limits():
    if Path(DISEASE_LIMIT_PATH).exists():
        with open(DISEASE_LIMIT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

def find_disease_info(name):
    limits = load_disease_limits()
    name_normalized = normalize_disease_name(name)
    for item in limits:
        if normalize_disease_name(item["diseaseName"]) == name_normalized:
            return {
                "avoid": [],
                "safe": [],
                "nutrition_limit": {
                    "protein": item.get("proteinLimit", 0),
                    "carbohydrates": item.get("sugarLimit", 0),
                    "fat": 0
                },
                "note": item.get("notes", "")
            }
    return {
        "avoid": [],
        "safe": [],
        "nutrition_limit": {},
        "note": "정보 없음"
    }

def process_disease(disease_name):
    return find_disease_info(disease_name)
