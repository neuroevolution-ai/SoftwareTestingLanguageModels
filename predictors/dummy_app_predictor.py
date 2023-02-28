from typing import Dict

from predictors.predictor import Predictor


class DummyAppPredictor(Predictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.env_config["type"] == "DummyApp", "DummyAppPredictor must be used with the DummyApp environment"
        assert len(self.prompt_templates) > 0, "This class must define at least one prompt template"

    @property
    def prompt_templates(self) -> Dict[int, str]:
        return {
            0: ("Your task is to execute specific commands by interacting with the GUI of a software application. "
                "Your response must only contain a single number between 0 (including) and 24 (also including), "
                "since the application consists of exactly 25 numbered buttons in total and your response "
                "indicates which corresponding button to click. The State shows which buttons have already been "
                "pressed and each button can only be clicked once. This is the State:"),
            1: "Your task is to select an integer between 0 and 24, which is not present in the following list:"
        }

    def convert_to_prompt(self, state) -> str:
        if self.current_prompt_id == 1:
            state = state["pressed_buttons"]

        prompt = f"{self.current_prompt_template} {state} Possible Actions: 0 to 24. Your Action:"

        return prompt
