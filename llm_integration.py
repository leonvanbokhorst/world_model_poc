# world_model_poc/llm_integration.py
import os
from litellm import completion


def get_goal_inference(observations):
    observation_text = "\n".join(f"- {obs}" for obs in observations)

    prompt = (
        "Based on the following sequence of actions, what is the most likely goal of the agent? "
        "The possible goals are: 'unlock the box', 'sit on the chair', or 'inspect the table'. "
        "Please state only the most likely goal, without any extra explanation.\n\n"
        f"Actions:\n{observation_text}\n\n"
        "Most likely goal:"
    )

    print(f"   - Sending Prompt to LLM:\n---\n{prompt}\n---")

    try:
        response = completion(
            model="ollama/llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in inferring agent intentions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=20,
        )
        raw_response = response.choices[0].message.content
        print(f"   - Received Raw Response from LLM: '{raw_response}'")
        # The response object structure from litellm is the same as OpenAI's
        return raw_response.strip()
    except Exception as e:
        # It's helpful to print the error to see what's going on with the local model
        print(f"Error calling local LLM: {e}")
        return f"Error inferring goal: {e}"
