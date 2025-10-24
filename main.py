# world_model_poc/main.py
import time
from .environment import Room
from .agents import TargetAgent, LearningAgent


def run_simulation():
    print("--- Starting World Model POC Simulation ---")

    # 1. Initialize Environment and Agents
    room = Room()
    target_agent = TargetAgent(goal="unlock the box")
    learning_agent = LearningAgent()

    print("\n--- Simulation Begins ---")

    # 2. Simulation Loop
    step = 1
    while True:
        print("\n" + "=" * 40)
        print(f"SIMULATION STEP {step}")
        print("=" * 40)

        # TargetAgent acts
        print(f"\n> TargetAgent's turn...")
        action = target_agent.get_next_action()
        if action is None:
            print("  TargetAgent has completed its actions. Simulation ending.")
            break

        print(f"  TargetAgent performs action: '{action}'")

        # Environment updates and generates an observation
        observation = room.update_state(action)

        # LearningAgent observes and updates its world model
        learning_agent.observe(observation)

        time.sleep(2)  # Pause for readability
        step += 1

    print("\n" + "=" * 40)
    print("SIMULATION REPORT")
    final_belief = learning_agent.world_model["target_agent_goal_belief"]
    actual_goal = target_agent.goal
    print(f"Final Inferred Goal: {final_belief}")
    print(f"Actual Goal: {actual_goal}")
    # Strip quotes and periods for a more robust comparison
    if final_belief.lower().strip(" '.\"") == actual_goal:
        print(
            "\nConclusion: The LearningAgent successfully inferred the TargetAgent's goal."
        )
    else:
        print(
            "\nConclusion: The LearningAgent did not correctly infer the TargetAgent's goal."
        )
    print("=" * 40)


if __name__ == "__main__":
    run_simulation()
