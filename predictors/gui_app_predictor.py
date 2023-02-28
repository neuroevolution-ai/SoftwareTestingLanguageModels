from typing import Dict

from predictors.predictor import Predictor


class GUIAppPredictor(Predictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.env_config["type"] == "GUIApp", "GUIAppPredictor must be used with the GUIApp environment"
        assert len(self.prompt_templates) > 0, "This class must define at least one prompt template"

    @property
    def prompt_templates(self) -> Dict[int, str]:
        return {
            0: ("Your task is to execute specific commands by interacting with the GUI of a software application. "
                "Your response must only contain an entry from a given list of clickable widgets "
                "since your response indicates which widget to click. This is the current state of the "
                "application:")
        }

    def convert_to_prompt(self, state) -> str:
        prompt = f"{self.current_prompt_template} {state}. Your Action:"

        return prompt
