# world_model_poc/main.py
import time
import random
import io
import sys
from contextlib import redirect_stdout
from .environment import Room
from .agents import TargetAgent, LearningAgent
from .llm_integration import get_meta_analysis, get_final_verdict


class Tee:
    """A helper class to write to multiple streams at once."""

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.stream1.flush()
        self.stream2.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()


def run_simulation():
    # Use StringIO to capture the log and Tee to stream to both console and the log
    log_stream = io.StringIO()
    tee_stream = Tee(sys.stdout, log_stream)

    with redirect_stdout(tee_stream):
        print("--- Starting World Model POC Simulation ---")

        # 1. Initialize Environment and Agents
        room = Room()
        env_description = room.get_state_description()

        # Add a new challenge: The target goal is now randomized
        possible_goals = ["unlock the box", "read the book"]
        chosen_goal = random.choice(possible_goals)

        target_agent = TargetAgent(goal=chosen_goal)
        learning_agent = LearningAgent(env_description=env_description)

        print(f"\nTargetAgent has been assigned a random, hidden goal.")

        print("\n--- Simulation Begins ---")

        # 2. Simulation Loop
        step = 1
        while True:
            print("\n" + "=" * 40)
            print(f"SIMULATION STEP {step}")
            print("=" * 40)

            # TargetAgent acts dynamically based on the current world state
            print(f"\n> TargetAgent's turn (Goal: '{target_agent.goal}')...")
            current_env_desc = room.get_state_description()
            action = target_agent.get_next_action(current_env_desc, room.objects)

            if action is None:
                print(
                    "  TargetAgent has completed its goal or reached its action limit. Simulation ending."
                )
                break

            print(f"  TargetAgent dynamically chose action: '{action}'")

            # Environment updates and generates an observation
            observation = room.update_state(action)

            # LearningAgent observes and updates its world model
            learning_agent.observe(observation)

            # Pause for user input to allow for step-by-step discussion
            try:
                input("\n... press Enter to continue to the next step ...")
            except EOFError:
                # In non-interactive environments, continue automatically.
                pass

            step += 1

        print("\n" + "=" * 40)
        print("SIMULATION REPORT")
        final_belief = learning_agent.world_model["target_agent_goal_belief"]
        actual_goal = target_agent.goal
        print(f"Final Inferred Goal: {final_belief}")
        print(f"Actual Goal: {actual_goal}")

        # Use the LLM to semantically compare the goals for a more robust conclusion
        print("\nChecking conclusion with LLM...")
        is_success = get_final_verdict(final_belief, actual_goal)

        if is_success:
            print(
                "\nConclusion: The LearningAgent successfully inferred the TargetAgent's goal."
            )
        else:
            print(
                "\nConclusion: The LearningAgent did not correctly infer the TargetAgent's goal."
            )
        print("=" * 40)

    # The simulation has been streamed live, so we don't need to print the log again.
    # We just need to get its value for the analysis.
    simulation_log = log_stream.getvalue()

    # 3. Perform LLM Meta-Analysis
    print("\n" + "=" * 50)
    print("LLM META-ANALYSIS OF SIMULATION")
    print("=" * 50)
    # The get_meta_analysis function handles streaming the output directly
    get_meta_analysis(simulation_log)
    print("=" * 50)


if __name__ == "__main__":
    run_simulation()
