
# reward_server.py
from flask import Flask, request, jsonify
import boto3, json

app = Flask(__name__)
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

@app.route("/reward", methods=["POST"])
def reward():
    data = request.json

    # Support two formats:
    # 1. Legacy format: {"prompt": "...", "system": "..."}
    # 2. New format: {"content": [...], "system": "..."}

    # Get optional system prompt
    system_prompt = data.get("system")

    # Get max_tokens from request, default to 4000 if not provided
    max_tokens = data.get("max_tokens", 4000)

    # Get temperature from request, default to 0
    temperature = data.get("temperature", 0)

    # Get top_p from request, default to 0.99
    top_p = data.get("top_p", 0.99)

    # Determine content based on format
    if "content" in data:
        # New format: content is already formatted as content blocks
        content = data["content"]
    elif "prompt" in data:
        # Legacy format: prompt can be string or list
        prompt = data["prompt"]
        if isinstance(prompt, str):
            content = prompt
        elif isinstance(prompt, list):
            # Multimodal format - prompt is already formatted as content blocks
            content = prompt
        else:
            content = str(prompt)
    else:
        # Fallback - try to extract from data
        return jsonify({"error": "Missing 'content' or 'prompt' field in request"}), 400

    # Build the request body for Bedrock
    body = {
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": temperature,
        "top_p": top_p
    }

    # Add system prompt if provided separately
    if system_prompt:
        body["system"] = system_prompt

    resp = bedrock.invoke_model(
        body=json.dumps(body),
        modelId="us.anthropic.claude-sonnet-4-20250514-v1:0"
    )
    result = json.loads(resp["body"].read())
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
