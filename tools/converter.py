import json
from typing import Optional

import openai
from transformers import AutoTokenizer, AutoModelForCausalLM


class Predictor:

    def __init__(self, model_name: str, use_openai: bool, max_new_tokens: int, temperature: float):
        self.model_name = model_name
        self.use_openai = use_openai
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if self.use_openai:
            with open("tools/openai_key.json", "r") as file:
                openai.api_key = json.load(file)["key"]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Set pad_token_id manually to fix HuggingFace warning, see https://stackoverflow.com/a/71397707
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                pad_token_id=self.tokenizer.eos_token_id
            )

            if temperature == 0:
                raise RuntimeError(f"A temperature of 0 is only supported with OpenAI, try slightly higher values "
                                   "than 0.")

    @staticmethod
    def convert_to_prompt(task, pressed_buttons):
        meta_info = ("Your task is to execute specific commands by interacting with the GUI of a software application. "
                     "Your response must only contain a single number between 0 (including) and 24 (also including), "
                     "since the application consists of exactly 25 numbered buttons in total and your response "
                     "indicates which corresponding button to click. The State shows which buttons have already been "
                     "pressed and each button can only be clicked once. This is the command: ")

        # TODO 0-24 actions specific to dummyapp -> must be generalized
        prompt = f"{meta_info} {task} State: {pressed_buttons} Possible Actions: 0 to 24 Your Action:"

        # prompt = ("This is a list of already clicked buttons numbered from 0 and 24: "
        #           f"{pressed_buttons['pressed buttons']}. Guess the number of a remaining unclicked button by "
        #           "specifying a single integer between 0 and 24:")

        return prompt

    def predict(self, prompt) -> Optional[int]:
        if self.use_openai:
            prediction = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature
            )

            raw_prediction = prediction["choices"][0]["text"]
        else:
            encoded_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids

            encoded_model_prediction = self.model.generate(
                encoded_prompt,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens
            )

            # From the prediction, use only the predicted tokens (self.max_new_tokens), decode them, then discard
            # the batch dimension with "[0]"
            raw_prediction = self.tokenizer.batch_decode(encoded_model_prediction[:, -self.max_new_tokens:])[0]

        try:
            button_to_press = int(raw_prediction)
        except ValueError:
            button_to_press = None

        return button_to_press
