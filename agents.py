# world_model_poc/agents.py
from .llm_integration import get_goal_inference

class TargetAgent:
    def __init__(self, goal):
        self.goal = goal
        self.actions = self._plan_actions()
        print(f"TargetAgent initialized with a hidden goal.")

    def _plan_actions(self):
        if self.goal == "unlock the box":
            return ["pick up key", "unlock box"]
        elif self.goal == "sit on the chair":
            return ["move to chair", "sit down"]
        else:
            return ["look around"]

    def get_next_action(self):
        if self.actions:
            return self.actions.pop(0)
        return None

class LearningAgent:
    def __init__(self):
        self.world_model = {
            "target_agent_goal_belief": "unknown",
            "observation_history": []
        }
        print("LearningAgent initialized with an empty world model.")

    def observe(self, observation):
        self.world_model["observation_history"].append(observation)
        print(f"LearningAgent observes: \"{observation}\"")
        self.update_mental_model()

    def update_mental_model(self):
        print("LearningAgent is updating its mental model...")
        inferred_goal = get_goal_inference(self.world_model["observation_history"])
        
        # Predictive Coding: The "error" is the change in belief.
        if inferred_goal and inferred_goal != self.world_model["target_agent_goal_belief"]:
            print(f"Prediction error detected! Belief updated.")
            self.world_model["target_agent_goal_belief"] = inferred_goal
        
        print(f"LearningAgent's current belief about TargetAgent's goal: {self.world_model['target_agent_goal_belief']}")
