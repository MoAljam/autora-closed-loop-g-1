from abc import ABC, abstractmethod
from typing import List, TypeVar, Union

from sweetbean.parameter import (
    DataVariable,
    DerivedLevel,
    DerivedParameter,
    TimelineVariable,
    param_to_psych,
)

from sweetbean.stimulus import Stimulus
from sweetbean.sequence import Block, Timeline
from sweetbean.parameter import CodeVariable
from sweetbean.stimulus import StimulusVar, TimelineVariable, SurveyStimulus
from sweetbean.update_package_honeycomb import get_import, update_package

import re


def update_html_script(file_path, target_path=None):
    # Read the original HTML content
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Define the JavaScript replacement text
    js_replacement = r"""
    <script>
    jsPsych = initJsPsych({
        on_finish: function () {
            jsPsych.data.displayData();
            // download the data
            jsPsych.data.get().localSave('csv', 'myexperiment.csv');
        }
    });"""

    # Use regular expression to find the <script> block and replace it
    updated_content = re.sub(r"<script>\s*jsPsych\s*=\s*initJsPsych\(\);", js_replacement, content, flags=re.DOTALL)

    # Write the updated HTML back to the file
    if target_path is None:
        target_path = file_path

    with open(target_path, "w", encoding="utf-8") as file:
        file.write(updated_content)


StringType = Union[None, str, DerivedParameter, TimelineVariable]
IntType = Union[None, int, TimelineVariable, DerivedParameter]
FloatType = Union[None, float, TimelineVariable, DerivedParameter]
StringTypeL = Union[List[StringType], StringType]
IntTypeL = Union[List[IntType], IntType]

HTML_PREAMBLE = (
    "<!DOCTYPE html>\n"
    "<head>\n"
    "<title>My awesome experiment</title>"
    '<script src="https://unpkg.com/jspsych@7.3.1"></script>\n'
    '<script src="https://unpkg.com/@jspsych/plugin-html-keyboard-response@1.1.2"></script>\n'
    '<script src="https://unpkg.com/@jspsych/plugin-survey-text@1.1.2"></script>\n'
    '<script src="https://unpkg.com/@jspsych/plugin-survey-multi-choice@1.1.2"></script>\n'
    '<script src="https://unpkg.com/@jspsych/plugin-survey-likert@1.1.2"></script>\n'
    '<script src="https://unpkg.com/@jspsych/plugin-survey-likert@1.1.2"></script>\n'
    '<link href="https://unpkg.com/jspsych@7.3.1/css/jspsych.css" rel="stylesheet"'
    ' type="text/css"/>\n'
    '<script src="https://unpkg.com/@jspsych/plugin-image-keyboard-response@1.1.2"></script>\n'
    '<script src="https://unpkg.com/@jspsych/plugin-video-keyboard-response@1.1.2"></script>\n'
    # '<script src="https://unpkg.com/@jspsych-contrib/plugin-rok@1.1.1"></script>'
    '<script src="./index.browser.min.js"></script>'
    "<style>"
    "body {"
    "background: #000; color: #fff;"
    "}\n"
    "div {"
    "font-size:36pt; line-height:40pt"
    "}"
    ".sweetbean-square {"
    "width:10vw; height:10vw"
    "}"
    ".sweetbean-circle {"
    "width:10vw; height:10vw; border-radius:50%"
    "}"
    ".sweetbean-triangle {"
    "width:10vw; height:10vw; clip-path:polygon(50% 0, 0 100%, 100% 100%)"
    "}"
    ".feedback-screen-red {"
    "position:absolute; left:0; top:0; width:100vw; height: 100vh; background: red"
    "}"
    ".feedback-screen-green {"
    "position:absolute; left:0; top: 0; width:100vw; height: 100vh; background: green"
    "}"
    "</style>\n"
    "</head>\n"
    "<body></body>\n"
    "<script>\n"
)
HTML_APPENDIX = "</script>\n" "</html>"


def FUNCTION_PREAMBLE(is_async):
    async_string = ""
    if is_async:
        async_string = "async "
    return f"{async_string}function runExperiment() " + "{\n"


def FUNCTION_APPENDIX(is_async):
    async_string = ""
    if is_async:
        async_string = "await "
    return (
        f"{async_string}jsPsych.run(trials)\nconst observation = jsPsych.data.get()\n"
        + f"return {async_string}observation\n"
        + "}"
    )


def TEXT_APPENDIX(is_async):
    async_string = ""
    if is_async:
        async_string = "await "
    return f"{async_string}jsPsych.run(trials)\n"


# class text_survey_stimulus(SurveyStimulus):
#     def __init__(self,
#                  prompts=[],
#                  placeholder: StringTypeL="0",
#                  size: IntTypeL=1):
#         type = "jsPsychSurveyText"
#         super().__init__(locals())

#     def _stimulus_to_psych(self):
#         self.text_trial += self._set_param_js_preamble("questions")
#         self.text_trial += self._set_set_variable("prompts")
#         self.text_trial += "\nlet prompts_ = []"
#         self.text_trial += (
#             f'\nfor (const p of {self._set_get_variable("prompts")})' + "{"
#         )
#         self.text_trial += "\nprompts_.push({'prompt': p,"
#         self.text_trial += " 'placeholder': "
#         self.text_trial += {self._set_get_variable("placeholder")}
#         self.text_trial += " 'size': "
#         self.text_trial += {self._set_get_variable("size")}
#         self.text_trial += " })}"
#         self.text_trial += "return prompts_},"
#         self._set_data_text("prompts")


class rdp_rsvp_stimulus(Stimulus):
    """
    using kinetogram and textvisual stimulus
    """

    def __init__(
        self,
        duration: Union[None, int, TimelineVariable, DerivedParameter] = None,
        number_of_oobs: IntTypeL = 300,
        number_of_apertures: IntType = 1,
        coherent_movement_direction: IntTypeL = None,
        coherent_orientation: IntTypeL = None,
        coherence_movement: IntTypeL = 100,
        coherence_orientation: IntTypeL = 100,
        movement_speed: IntTypeL = 2,
        aperture_width: IntTypeL = 300,
        aperture_height: IntTypeL = 300,
        aperture_position_left: IntTypeL = 50,
        aperture_position_top: IntTypeL = 50,
        aperture_shape: IntTypeL = 1,
        fade_out: IntTypeL = 0,
        oob_color: StringTypeL = "white",
        background_color: StringType = "grey",
        stimulus_type: IntTypeL = 0,
        text: StringType = "S",
        prompt: StringType = "S",
        color: StringType = "black",
        choices: List[str] = ["NO_KEYS"],
        correct_key: StringType = "",
        correct_choice: List[str]= [""],
    ):
        type = "jsPsychRok"
        # type = "jsPsychHtmlKeyboardResponse"
        super().__init__(locals())

    def _stimulus_to_psych(self):
        self.text_trial += self._set_param_js_preamble("stimulus")
        self.text_trial += self._set_set_variable("text")
        self.text_trial += self._set_set_variable("prompt")
        self.text_trial += self._set_set_variable("color")
        self.text_trial += "return "
        self.text_trial += (
            f'"<div style=\'color: "+{self._set_get_variable("color")}+"\'>"'
            f'+{self._set_get_variable("text")}+"</div>"'
            "},"
        )
        self._set_param_full("text")
        self._set_param_full("prompt")
        self._set_param_full("color")
        self._set_param_full("number_of_oobs")
        self._set_param_full("number_of_apertures")
        self._set_param_full("coherent_movement_direction")
        self._set_param_full("coherent_orientation")
        self._set_param_full("coherence_movement")
        self._set_param_full("coherence_orientation")
        self._set_param_full("oob_color")
        self._set_param_full("background_color")
        self._set_param_full("movement_speed")
        self._set_param_full("aperture_width")
        self._set_param_full("aperture_height")
        self._set_param_full("aperture_position_left")
        self._set_param_full("aperture_position_top")
        self._set_param_full("aperture_shape")
        self._set_param_full("fade_out")
        # self._set_param_full("choices")
        self._set_param_full("correct_choice")
        self._set_param_full("stimulus_type")
        self.text_trial += self._set_param_js_preamble("correct_choice")
        self.text_trial += self._set_set_variable("correct_key")
        self.text_trial += "return [correct_key] },"

    def _correct_to_psych(self):
        if "correct_key" in self.arg:
            self._set_data_text("correct_key")
            self.text_data += self._set_set_variable("correct")
            self.text_data += 'data["bean_correct"] = data["correct"]'


class Experiment:
    blocks: List[Block] = []
    text_js = ""

    def __init__(self, blocks: List[Block]):
        self.blocks = blocks
        self.to_psych()

    def to_psych(self):
        self.text_js = "jsPsych = initJsPsych();\n"
        self.text_js += "trials = [\n"
        for b in self.blocks:
            self.text_js += b.text_js
            self.text_js += ","
        self.text_js = self.text_js[:-1] + "]\n"
        self.text_js += ";jsPsych.run(trials)"

    def to_html(self, path):
        html = HTML_PREAMBLE
        # print(html)
        blocks = 0
        for b in self.blocks:
            if b.timeline and isinstance(b.timeline, Timeline):
                html += f'</script><script src="{b.timeline.path}">\n'
                blocks += 1
        if blocks > 0:
            html += "</script><script>\n"
        html += f"{self.text_js}" + HTML_APPENDIX

        with open(path, "w") as f:
            f.write(html)

    def to_js_string(
        self,
        as_function=False,
        is_async=False,
    ):
        text = FUNCTION_PREAMBLE(is_async) if as_function else ""
        text += "const jsPsych = initJsPsych()\n"
        text += "const trials = [\n"
        for b in self.blocks:
            text += b.text_js
            text += ","
        text = text[:-1] + "]\n"
        text += FUNCTION_APPENDIX(is_async) if as_function else TEXT_APPENDIX(is_async)
        return text
