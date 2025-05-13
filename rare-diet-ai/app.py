from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 프론트엔드 연동을 위한 CORS 허용

@app.route("/hello", methods=["POST"])
def say_hello():
    try:
        data = request.json
        name = data.get("name", "anonymous")
        return jsonify({
            "message": f"Hello, {name}!"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
