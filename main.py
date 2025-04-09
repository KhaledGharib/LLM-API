from flask import Flask, request, jsonify
import requests
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def extract_json_like_block(text):
    """Try to find and fix a valid JSON object in a text block, even if it's incomplete."""
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
                if not stack and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass

    if start_idx is not None:
        candidate = text[start_idx:]
        unclosed = candidate.count('{') - candidate.count('}')
        candidate += '}' * unclosed
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None

@app.route('/request', methods=['POST'])
def send_to_ollama():
    data = request.json
    user_message = data.get('request')

    if not user_message:
        return jsonify({"error": "User message is required"}), 400

    # 🔧 Updated prompt with device argument included
    system_prompt = """
You are a smart home assistant. When the user gives a command, respond with a JSON describing the function to call and the arguments.

Always include:
- "room": which room the command refers to
- "device": the specific device to control (like "lamp", "fan", "AC")
- "command": what to do (e.g., "on", "off", "set") can't be left empty
- "function": [turn_on, turn_off]


User: Turn off the bedroom lamp.

Assistant:
{
  "function": "turn_off",
  "arguments": {
    "room": "bedroom",
    "device": "lamp",
    "command": "off"
  }
}

User: Set the living room fan to on.

Assistant:
{
  "function": "turn_on",
  "arguments": {
    "room": "living room",
    "device": "fan",
    "command": "on"
  }
}
User: Switch off the kitchen light.

Assistant:
{
  "function": "turn_off",
  "arguments": {
    "room": "kitchen",
    "device": "light",
    "command": "off"
  }
}

User: Please turn on the AC in the living room.

Assistant:
{
  "function": "turn_on",
  "arguments": {
    "room": "living room",
    "device": "AC",
    "command": "on"
  }
}

User: Shut down the fan in the guest room.

Assistant:
{
  "function": "turn_off",
  "arguments": {
    "room": "guest room",
    "device": "fan",
    "command": "off"
  }
}

User: Power on the heater in the bathroom.

Assistant:
{
  "function": "turn_on",
  "arguments": {
    "room": "bathroom",
    "device": "heater",
    "command": "on"
  }
}

User: Turn off the light in the hallway.

Assistant:
{
  "function": "turn_off",
  "arguments": {
    "room": "hallway",
    "device": "light",
    "command": "off"
  }
}

User: Activate the fan in the master bedroom.

Assistant:
{
  "function": "turn_on",
  "arguments": {
    "room": "master bedroom",
    "device": "fan",
    "command": "on"
  }
}

User: Kill the lamp in the study room.

Assistant:
{
  "function": "turn_off",
  "arguments": {
    "room": "study room",
    "device": "lamp",
    "command": "off"
  }
}

User: Fire up the AC in the kids' room.

Assistant:
{
  "function": "turn_on",
  "arguments": {
    "room": "kids' room",
    "device": "AC",
    "command": "on"
  }
}

User: Lights off in the dining room.

Assistant:
{
  "function": "turn_off",
  "arguments": {
    "room": "dining room",
    "device": "light",
    "command": "off"
  }
}

User: I want the heater turned on in the office.

Assistant:
{
  "function": "turn_on",
  "arguments": {
    "room": "office",
    "device": "heater",
    "command": "on"
  }
}

"""

    payload = {
        "model": "smollm2:360m",  # Replace if needed
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "stream": False
    }

    try:
        response = requests.post('http://127.0.0.1:11434/api/chat', json=payload)
        response_json = response.json()
    except Exception as e:
        return jsonify({
            "error": "Error communicating with Ollama",
            "details": str(e)
        }), 500

    content = response_json.get('message', {}).get('content', '')
    json_obj = extract_json_like_block(content)

    if json_obj:
        return jsonify(json_obj), 200
    else:
        return jsonify({
            "error": "Failed to parse JSON from model response",
            "raw": content
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
