# world_model_poc/agents.py
from .llm_integration import (
    get_goal_inference,
    get_final_verdict,
    get_next_action_from_llm,
)


class TargetAgent:
    def __init__(self, goal):
        self.goal = goal
        self.action_history = []
        print(f"TargetAgent initialized with a dynamic, LLM-driven goal: '{goal}'.")

    def is_goal_achieved(self, env_objects):
        """Check if the goal has been met based on the state of the world."""
        if self.goal == "unlock the box" and env_objects.get("box") == "unlocked":
            print("  TargetAgent confirms its goal ('unlock the box') is complete.")
            return True
        if self.goal == "read the book":
            # This is a bit trickier, we'll say sitting down with the book achieves it.
            if (
                env_objects.get("book") == "in TargetAgent's possession"
                and "sit on chair" in self.action_history
            ):
                print("  TargetAgent confirms its goal ('read the book') is complete.")
                return True
        return False

    def get_next_action(self, environment_description, env_objects):
        if self.is_goal_achieved(env_objects):
            return None

        # Limit the history to avoid overly long prompts
        if len(self.action_history) > 5:
            print("  TargetAgent has reached its action limit. Ending turn.")
            return None

        print("  TargetAgent is thinking about its next action...")
        next_action = get_next_action_from_llm(
            self.goal, environment_description, self.action_history
        )
        self.action_history.append(next_action)
        return next_action


class LearningAgent:
    def __init__(self, env_description):
        self.world_model = {
            "environment_description": env_description,
            "target_agent_goal_belief": "unknown",
            "observation_history": [],
        }
        print("LearningAgent initialized and has received the room description.")

    def observe(self, observation):
        self.world_model["observation_history"].append(observation)
        print(f'> LearningAgent observes: "{observation}"')
        self.update_mental_model()

    def update_mental_model(self):
        print("\n--- LearningAgent Reasoning Cycle ---")
        prior_belief = self.world_model["target_agent_goal_belief"]

        print(f"1. Prior Belief: '{prior_belief}'")

        print("2. Reviewing Observation History:")
        for obs in self.world_model["observation_history"]:
            print(f'   - "{obs}"')

        print("3. Querying LLM for new inference...")
        inferred_goal = get_goal_inference(
            self.world_model["observation_history"],
            self.world_model["environment_description"],
        )

        # Predictive Coding: Use an LLM to semantically check if the belief has truly changed.
        beliefs_are_different = not get_final_verdict(inferred_goal, prior_belief)

        if inferred_goal and beliefs_are_different:
            print(f"4. Prediction Error Detected! (Belief has semantically changed)")
            print(f"   - Old Belief: '{prior_belief}'")
            print(f"   - New Inference: '{inferred_goal}'")
            print("5. Updating World Model...")
            self.world_model["target_agent_goal_belief"] = inferred_goal
        else:
            print("4. No significant semantic change in belief. World Model is stable.")

        print(f"6. Current Belief: '{self.world_model['target_agent_goal_belief']}'")
        print("-----------------------------------")
