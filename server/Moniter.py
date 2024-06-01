from flask import Flask, request, jsonify
import json

# 假设 RAG 是你已经定义好的模块，并且有 ai 对象和相应的方法
from RAG import *

app = Flask(__name__)
ai = RAG()


# 跨域处理函数
def cross_domain(
    origin=None,
    methods=None,
    headers=None,
    expose_headers=None,
    max_age=21600,
    credentials=False,
    content_type=None,
):
    if headers is not None and not isinstance(headers, list):
        headers = [headers]
    if methods is not None and not isinstance(methods, list):
        methods = [methods]
    if expose_headers is not None and not isinstance(expose_headers, list):
        expose_headers = [expose_headers]

    response_headers = {
        "Access-Control-Allow-Origin": origin if origin else "*",
        "Access-Control-Allow-Methods": (
            ", ".join(methods) if methods else "GET, POST, OPTIONS"
        ),
        "Access-Control-Allow-Headers": (
            ", ".join(headers) if headers else "Content-Type"
        ),
        "Access-Control-Max-Age": str(max_age),
        "Access-Control-Allow-Credentials": "true" if credentials else "false",
    }

    if content_type is not None:
        response_headers["Content-Type"] = content_type

    return response_headers


@app.after_request
def apply_cross_domain(response):
    response.headers.extend(cross_domain())
    return response


@app.route("/", methods=["POST", "OPTIONS"])
def handle_post():
    if request.method == "OPTIONS":
        # 对于预检请求，直接返回200 OK
        return jsonify({"message": "CORS preflight successful"})

    try:
        data = request.get_json()
        question = data.get("question")
        respond = ai.answer(ai.prompt(question))
        return respond, 200
    except Exception as err:
        return jsonify({"error": "Error, 请重试 " + str(err)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
