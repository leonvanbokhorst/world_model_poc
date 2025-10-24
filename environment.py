# world_model_poc/environment.py


class Room:
    def __init__(self):
        self.objects = {
            "key": "on the table",
            "box": "locked",
            "chair": "near the table",
            "table": "in the center of the room",
        }
        print(
            "Environment: A room with a table, a chair, a key on the table, and a locked box."
        )

    def get_state_description(self):
        descriptions = []
        for obj, state in self.objects.items():
            if "on" in state or "near" in state or "in" in state:
                descriptions.append(f"The {obj} is {state}.")
            elif state == "locked":
                descriptions.append(f"The box is {state}.")
            elif state == "unlocked":
                descriptions.append(f"The box is {state}.")
        return " ".join(descriptions)

    def update_state(self, action):
        if action == "pick up key":
            if self.objects["key"] == "on the table":
                self.objects["key"] = "in TargetAgent's possession"
                return "TargetAgent picked up the key from the table."
            else:
                return "The key is not on the table."
        elif action == "unlock box":
            if self.objects["key"] == "in TargetAgent's possession":
                if self.objects["box"] == "locked":
                    self.objects["box"] = "unlocked"
                    return "TargetAgent used the key to unlock the box."
                else:
                    return "The box is already unlocked."
            else:
                return "TargetAgent does not have the key."
        return f"TargetAgent performs an unknown action: {action}"
