import json
import os
from typing import Union, List

import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

SEQ2SEQ_MODELS = ["google/flan-t5-base", "allenai/tk-instruct-3b-def"]


class Predictor:

    def __init__(self, model_name: str, use_openai: bool, max_new_tokens: int, num_return_sequences: int,
                 temperature: float, do_sample: bool):
        self.model_name = model_name
        self.use_openai = use_openai
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature  # TODO implement increasing temperature when LM comes to a halt
        self.do_sample = do_sample

        if self.use_openai:
            if os.getenv("OPENAI_API_KEY") is None:
                with open("tools/openai_key.json", "r") as file:
                    openai.api_key = json.load(file)["key"]
        else:
            if self.do_sample and temperature == 0:
                raise RuntimeError(f"A temperature of 0 is only supported with OpenAI or if do_sample=False, try "
                                   "slightly higher values than 0.")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            print("Loading HuggingFace model...")

            # Some models use the text generation task, other the text2text task. For the latter, the seq2seq model
            # class is needed, for the former the causal LM class
            if self.model_name in SEQ2SEQ_MODELS:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name
                )
            else:
                # Set pad_token_id manually to fix HuggingFace warning, see https://stackoverflow.com/a/71397707
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            print("Loading complete")

    @staticmethod
    def convert_to_prompt(state, meta_info_id: int):
        # TODO 0-24 actions specific to dummyapp -> must be generalized
        if meta_info_id == 0:
            meta_info = (
                "Your task is to execute specific commands by interacting with the GUI of a software application. "
                "Your response must only contain a single number between 0 (including) and 24 (also including), "
                "since the application consists of exactly 25 numbered buttons in total and your response "
                "indicates which corresponding button to click. The State shows which buttons have already been "
                "pressed and each button can only be clicked once. This is the State:"
            )
        else:
            meta_info = (
                "Your task is to select an integer between 0 and 24, which is not present in the following list:"
            )

            state = state["pressed buttons"]

        prompt = f"{meta_info} {state} Possible Actions: 0 to 24 Your Action:"

        return prompt

    def predict(self, prompt) -> List[Union[int, str]]:
        if self.use_openai:
            prediction = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature
            )

            raw_predictions = [prediction["choices"][0]["text"]]
        else:
            encoded_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids

            encoded_model_prediction = self.model.generate(
                encoded_prompt,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_return_sequences,
                num_beams=self.num_return_sequences
            )

            # From the prediction, use only the predicted tokens (self.max_new_tokens), decode them, then discard
            # the batch dimension with "[0]"
            if self.model_name in SEQ2SEQ_MODELS:
                raw_predictions = self.tokenizer.batch_decode(encoded_model_prediction, skip_special_tokens=True)
            else:
                raw_predictions = self.tokenizer.batch_decode(encoded_model_prediction[:, -self.max_new_tokens:])

        possible_buttons_to_press = []

        for raw_pred in raw_predictions:
            try:
                button = int(raw_pred)
            except ValueError:
                possible_buttons_to_press.append(raw_pred)
            else:
                possible_buttons_to_press.append(button)

        return possible_buttons_to_press
