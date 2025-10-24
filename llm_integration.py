# world_model_poc/llm_integration.py
import os
from litellm import completion
import sys
from .config import OLLAMA_MODEL


def stream_and_capture(response_generator):
    """
    Streams a generator to stdout and captures the full response.
    """
    full_response = ""
    for chunk in response_generator:
        chunk_content = chunk.choices[0].delta.content or ""
        print(chunk_content, end="", flush=True)
        full_response += chunk_content
    print()  # for a newline after the stream
    return full_response


def get_goal_inference(observations, environment_description):
    observation_text = "\n".join(f"- {obs}" for obs in observations)

    prompt = (
        "You are an AI observing another agent in a room. Your task is to infer the other agent's high-level goal. "
        "First, here is the description of the room and its objects:\n"
        f"Room State: {environment_description}\n\n"
        "Now, here is the sequence of actions you have observed the agent perform:\n"
        f"{observation_text}\n\n"
        "Based on the room's contents and the agent's actions, what is the most likely high-level goal of the agent? "
        "State the goal clearly and concisely, without any extra explanation."
    )

    print(f"   - Sending Open-Ended Prompt to LLM:\n---\n{prompt}\n---")

    try:
        response_stream = completion(
            model="ollama/phi4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in inferring agent intentions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=20,
            stream=True,
        )
        print(f"   - Streaming Response from LLM: ", end="", flush=True)
        full_response = stream_and_capture(response_stream)
        return full_response.strip()
    except Exception as e:
        # It's helpful to print the error to see what's going on with the local model
        print(f"Error calling local LLM: {e}")
        return f"Error inferring goal: {e}"


def get_meta_analysis(simulation_log):
    prompt = (
        "You are a cognitive scientist analyzing the behavior of an AI agent. "
        "The following is a log from a simulation where a 'LearningAgent' observed a 'TargetAgent' to infer its hidden goal. "
        "Please analyze the LearningAgent's performance. Explain how its reasoning process demonstrates the principles of predictive coding. "
        "Point out where its initial beliefs were incorrect and identify the key observation that allowed it to correct its world model and find the right goal.\n\n"
        "--- SIMULATION LOG ---\n"
        f"{simulation_log}"
        "\n--- END OF LOG ---\n\n"
        "Your analysis:"
    )

    print("Querying LLM for meta-analysis of the simulation...")

    try:
        response_stream = completion(
            model="ollama/phi4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True,
        )
        print("\n--- LLM Analysis Stream ---\n")
        full_analysis = stream_and_capture(response_stream)
        print("\n--- End of Stream ---")
        return full_analysis.strip()
    except Exception as e:
        print(f"Error calling local LLM for meta-analysis: {e}")
        return f"Error generating meta-analysis: {e}"


def get_final_verdict(inferred_goal, actual_goal):
    prompt = (
        "Does the 'Inferred Goal' semantically match the 'Actual Goal'? "
        "Please answer with only the word 'Yes' or 'No'.\n\n"
        f'Inferred Goal: "{inferred_goal}"\n'
        f'Actual Goal: "{actual_goal}"'
    )

    try:
        response = completion(
            model=f"ollama/{OLLAMA_MODEL}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3,
        )
        verdict = response.choices[0].message.content.strip().lower()
        return "Yes" in verdict
    except Exception as e:
        print(f"Error getting final verdict: {e}")
        return False


def get_next_action_from_llm(goal, environment_description, action_history):
    history_text = (
        "\n".join(f"- {act}" for act in action_history)
        if action_history
        else "No actions taken yet."
    )

    # This list could be dynamically generated from the environment in a more advanced version
    possible_actions = [
        "inspect table",
        "pick up key",
        "pick up book",
        "unlock box",
        "sit on chair",
    ]

    prompt = (
        f"You are a decisive, goal-oriented AI agent. Your ONLY objective is: '{goal}'.\n"
        "Every action you take must be a logical step towards this goal. Do not get stuck in loops. If an action does not help, choose a different one.\n\n"
        f"CURRENT WORLD STATE:\n{environment_description}\n\n"
        f"ACTIONS TAKEN SO FAR:\n{history_text}\n\n"
        "Analyze the situation. From the list of possible actions below, what is the single most logical next action to take to achieve your goal?\n\n"
        f"Possible Actions: {', '.join(possible_actions)}\n\n"
        "Respond with only the single best action from the list."
    )

    try:
        # We give the planning agent a slightly higher temperature to encourage novel actions if stuck
        response = completion(
            model=f"ollama/{OLLAMA_MODEL}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=20,
        )
        action = response.choices[0].message.content.strip().lower()

        # Clean up the response to get just the action
        for valid_action in possible_actions:
            if valid_action in action:
                return valid_action
        return "look around"  # Default if no valid action is found
    except Exception as e:
        print(f"Error getting next action: {e}")
        return "look around"
