import openai
import json

# Get openai key from file
with open("tools/openai_key.json", "r") as file:
    openai.api_key = json.load(file)["key"]


def convert_to_prompt(task, pressed_buttons):

    prompt = task + "   State: " + str(pressed_buttons) + \
        "   Possible Actions: 0 to 24    Action:"

    return prompt


def predict(prompt):
    try:
        ai_prediction = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=1,
            temperature=0
        )
    except openai.error.InvalidRequestError:
        print("InvalidRequestError")

    button_to_press = int(ai_prediction['choices'][0]['text'])

    return button_to_press
