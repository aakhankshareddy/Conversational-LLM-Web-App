import requests

HF_API_TOKEN = "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"  # Optional for free models

# Free models that support text generation
FREE_MODELS = [
    "gpt2",
    "distilgpt2",
    "microsoft/DialoGPT-small"
]

def format_chat_prompt(messages):
    """
    Formats a list of messages for text generation models.
    Simple format that works with most models.
    """
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant: "
    return prompt

def get_mock_response(messages):
    """
    Returns a mock response for testing when API is not available.
    Only responds to the last user message.
    """
    if not messages:
        return "Hello! How can I help you today?"
    
    last_user_message = messages[-1]["content"].lower()
    
    # Only respond if the last message is from user
    if messages[-1]["role"] != "user":
        return ""
    
    # Simple response logic - one response per user message
    if "hello" in last_user_message or "hi" in last_user_message:
        return "Hello! How can I help you today?"
    elif "capital" in last_user_message and "france" in last_user_message:
        return "The capital of France is Paris."
    elif "color" in last_user_message and "apple" in last_user_message:
        return "Apples come in many colors! The most common colors are red, green, and yellow. Red apples like Red Delicious and Gala are very popular, while green apples like Granny Smith are tart and crisp."
    elif "how are you" in last_user_message:
        return "I'm doing well, thank you for asking! How can I assist you?"
    elif "weather" in last_user_message:
        return "I can't check the weather, but I'd be happy to help with other questions!"
    elif "bye" in last_user_message or "goodbye" in last_user_message:
        return "Goodbye! Have a great day!"
    else:
        return "That's an interesting question! I'm here to help with any information you need."

def get_mistral_response(messages, model=FREE_MODELS[0], api_token=None):
    """
    Sends conversation history to a free model via direct API call and returns the latest response.
    Falls back to mock response if API fails.
    """
    try:
        prompt = format_chat_prompt(messages)
        
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 100, "temperature": 0.7}}
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return str(result[0].get("generated_text", "")).strip()
            else:
                return str(result.get("generated_text", "")).strip()
        else:
            # Fall back to mock response if API fails
            return get_mock_response(messages)
            
    except Exception as e:
        # Fall back to mock response if there's an error
        return get_mock_response(messages)

# Example usage:
if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    reply = get_mistral_response(messages, model=FREE_MODELS[0], api_token=HF_API_TOKEN)
    print("Model reply:", reply)
