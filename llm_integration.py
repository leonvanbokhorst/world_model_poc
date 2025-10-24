# world_model_poc/llm_integration.py
import os
from openai import OpenAI
from .config import OPENAI_API_KEY

# It's better to use environment variables for API keys, but this is for the POC
if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
    print("Warning: OpenAI API key is not set. Please set it in config.py")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

def get_goal_inference(observations):
    if not client:
        return "LLM client not initialized. Cannot infer goal."

    observation_text = "\n".join(f"- {obs}" for obs in observations)
    
    prompt = (
        "Based on the following sequence of actions, what is the most likely goal of the agent? "
        "The possible goals are: 'unlock the box', 'sit on the chair', or 'inspect the table'. "
        "Please state only the most likely goal.\n\n"
        f"Actions:\n{observation_text}\n\n"
        "Most likely goal:"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in inferring agent intentions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=20,
            n=1,
            stop=None
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error inferring goal: {e}"
