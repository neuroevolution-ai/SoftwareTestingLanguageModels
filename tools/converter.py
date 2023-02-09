import openai
import json

# Get openai key from file
with open("tools/openai_key.json", "r") as file:
    openai.api_key = json.load(file)["key"]


def convert_to_prompt(task, pressed_buttons):

    meta_info = "Your task is to execute specific commands by interacting with the GUI of a software application. Your response must only contain a single number between 0 (including) and 24 (also including), since the application consists of exactly 25 numbered buttons in total and your response indicates which corresponding button to click. The State shows which buttons have already been pressed and each button can only be clicked once. This is the command:"
    prompt = meta_info + task + "   State: " + str(pressed_buttons) + \
        "   Possible Actions: 0 to 24    Action:"

    return prompt


def predict(prompt):
    try:
        ai_prediction = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=3,
            temperature=0
        )
    except openai.error.InvalidRequestError:
        print("InvalidRequestError")

    button_to_press = int(ai_prediction['choices'][0]['text'])

    return button_to_press
