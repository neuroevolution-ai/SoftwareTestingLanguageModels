import os
from datetime import datetime

import cv2
import numpy as np
from kivy.app import App
from kivy.config import Config
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from naturalnets.environments.i_environment import get_environment_class
from naturalnets.tools.utils import rescale_values

from predictors.predictor import Predictor

Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '440')

config1 = {
    "environment": {
        "type": "GUIApp",
        "number_time_steps": 200,
        "include_fake_bug": False
    }
}

config2 = {
    "environment": {
        "type": "DummyApp",
        "number_time_steps": 100,
        "screen_width": 400,
        "screen_height": 400,
        "number_button_columns": 5,
        "number_button_rows": 5,
        "button_width": 50,
        "button_height": 30,
        "fixed_env_seed": False,
        "force_consecutive_click_order": False
    }
}

config = config2


class MainLayout(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get environment class from configuration
        environment_class = get_environment_class(
            config["environment"]["type"])
        self.app = environment_class(config["environment"])
        self.app.reset()

        self.render_app()

        # self.predictor = Predictor(
        #     model_name="EleutherAI/gpt-neo-1.3B",
        #     use_openai=False,
        #     max_new_tokens=3,
        #     num_return_sequences=3,
        #     temperature=0.0,
        #     do_sample=False
        # )

        # HuggingFace instructional LM
        self.predictor = Predictor(
            model_name="google/flan-t5-base",
            use_openai=False,
            max_new_tokens=3,
            num_return_sequences=3,
            temperature=0.5,
            do_sample=False
        )

        # OpenAI example
        # self.predictor = Predictor(
        #     model_name="text-davinci-003",
        #     use_openai=True,
        #     max_new_tokens=3,
        #     num_return_sequences=1,
        #     temperature=0.0,
        #     do_sample=False
        # )

    def render_app(self):
        screen_size = self.app.get_screen_size()
        image = self.app.render_image()
        image = cv2.flip(image, 0)

        # Texture
        texture = Texture.create(
            size=(screen_size, screen_size), colorfmt='rgb')
        texture.blit_buffer(image.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.ids.image_container.texture = texture

    def touch_down(self, image, touch):
        if image.collide_point(touch.x, touch.y):

            # Absolute kivy coordinates
            image_position_abs = np.array([image.pos[0], image.pos[1]])
            touch_position_abs = np.array([touch.x, image.height-touch.y])

            # Relative widget coordinates
            center = np.array([image.width, image.height])/2
            touch_position = touch_position_abs + image_position_abs
            screen_origin = center - self.app.get_screen_size()/2
            current_action = touch_position - screen_origin

            observation = self.app.get_observation_dict()

            _, _, _, info = self.app.step(rescale_values(current_action,
                                                         previous_low=0,
                                                         previous_high=self.app.get_screen_size(),
                                                         new_low=-1,
                                                         new_high=1))

            self.render_app()

    def execute_command(self):

        task = self.ids.input_field.text
        allowed_actions = [i for i in range(25)]

        while True:
            observation = self.app.get_observation_dict()
            prompt = self.predictor.convert_to_prompt(pressed_buttons, meta_info_id=0)
            possible_buttons = self.predictor.predict(prompt)

            debug_output = f"Possible buttons: {possible_buttons}. Selected: "

            for button in possible_buttons:
                if button in allowed_actions and button not in observation["pressed buttons"]:
                    self.app.step_widget(button)
                    self.render_app()

                    debug_output += f"{button}"
                    break

            print(debug_output)

            # Take screenshot
            filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png"
            self.export_to_png(os.path.join("screenshots", filename))

            # Later the language model should make multiple actions per executed test case, remove this break statement then
            break

            if observation == self.app.get_observation_dict():
                print("Finished execution")
                break


class TestingApp(App):

    def build(self):
        self.title = 'Software Testing Tool with GPT'
        return MainLayout()


if __name__ == '__main__':
    TestingApp().run()
