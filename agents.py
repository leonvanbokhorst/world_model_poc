# world_model_poc/agents.py
from .llm_integration import get_goal_inference


class TargetAgent:
    def __init__(self, goal):
        self.goal = goal
        self.actions = self._plan_actions()
        print(f"TargetAgent initialized with a hidden goal.")

    def _plan_actions(self):
        if self.goal == "unlock the box":
            return ["inspect table", "pick up key", "unlock box"]
        elif self.goal == "read the book":
            return ["inspect table", "pick up book", "sit on chair"]
        elif self.goal == "sit on the chair":
            return ["sit on chair"]
        else:
            return ["look around"]

    def get_next_action(self):
        if self.actions:
            return self.actions.pop(0)
        return None


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

        # Predictive Coding: The "error" is the difference between prior belief and new inference.
        normalized_inferred = inferred_goal.lower().strip(" '.\"")
        normalized_prior = prior_belief.lower().strip(" '.\"")

        if inferred_goal and normalized_inferred != normalized_prior:
            print(f"4. Prediction Error Detected!")
            print(f"   - Old Belief: '{prior_belief}'")
            print(f"   - New Inference: '{inferred_goal}'")
            print("5. Updating World Model...")
            self.world_model["target_agent_goal_belief"] = inferred_goal
        else:
            print("4. No significant change in belief. World Model is stable.")

        print(f"6. Current Belief: '{self.world_model['target_agent_goal_belief']}'")
        print("-----------------------------------")
