from sweetbean.parameter import TimelineVariable
from sweetbean.sequence import Block
from sweetbean.stimulus import (
    TextStimulus,
    BlankStimulus,
    FeedbackStimulus,
    RandomDotPatternsStimulus,
    RandomObjectKinematogramStimulus,
)

from utils import rdp_rsvp_stimulus, Experiment


def stimulus_sequence(experiment_timeline, training_timeline):
    coherence_ratio = TimelineVariable("coherence_ratio", [])
    motion_direction = TimelineVariable("motion_direction", [])

    item_1 = TimelineVariable("item_1", [])
    item_2 = TimelineVariable("item_2", [])
    item_3 = TimelineVariable("item_3", [])
    item_4 = TimelineVariable("item_4", [])
    item_5 = TimelineVariable("item_5", [])
    item_6 = TimelineVariable("item_6", [])
    item_7 = TimelineVariable("item_7", [])
    item_8 = TimelineVariable("item_8", [])

    correct_response = TimelineVariable("correct_response", [])

    choices = TimelineVariable("choices", [])
    # introduction TODO write introduction
    introduction = TextStimulus(text="Welcome to this experiment!<br>...<br>Press SPACE to continue.", choices=[" "])

    # instruction TODO write instruction
    instruction = TextStimulus(text="... Press SPACE to continue.", choices=[" "])

    # training onboarding TODO write training onboarding
    training_boarding = TextStimulus(text="training trails:<br>... Press SPACE to start the training.", choices=[" "])

    # experiment onboarding TODO write experiment onboarding
    experiment_boarding = TextStimulus(
        text="experiment trails:<br>... Press SPACE to start the experiment.", choices=[" "]
    )

    # break TODO incorporate breaks after specific number of trials
    pause = TextStimulus(
        text="Feel free to take a short break now<br>Press SPACE when you are ready to continue the experiment.",
        choices=[" "],
    )

    # debriefing/closure TODO write debriefing
    debriefing = TextStimulus(
        text="... Thank you for your participation!<br>Press SPACE to end the experiment.", choices=[" "]
    )

    # fixation cross and blank screens around it
    fixation_onset = BlankStimulus(duration=680)
    fixation = TextStimulus(duration=915, text="+")
    fixation_offset = BlankStimulus(duration=400)

    # blank screen in between items within a trial
    between_items = BlankStimulus(duration=45)

    # participant response
    response = TextStimulus(
        text="Which two numbers did you see during the previously displayed sequence?", choices=["1", "2", "3", "4"]
    )

    # timeline variables
    # independent variables

    def rsvp_maker(item, coherence_ratio=coherence_ratio, motion_direction=motion_direction):
        rdp = rdp_rsvp_stimulus(
            duration=500,
            number_of_oobs=20,
            number_of_apertures=1,
            movement_speed=40,
            coherence_movement=coherence_ratio,
            coherent_movement_direction=motion_direction,
            oob_color="white",
            background_color="black",
            aperture_height=300,
            aperture_width=300,
            # choices   =choices,
            stimulus_type=1,  # 1 is for circles
            text=item,
            prompt=item,
            color="black",
        )
        return rdp

    # create all lists of sequences and individual blocks
    introduction_list = [introduction]
    introduction_block = Block(introduction_list)

    instruction_list = [instruction]
    instruction_block = Block(instruction_list)

    training_boarding_list = [training_boarding]
    training_boarding_block = Block(training_boarding_list)

    training_list = [
        fixation_onset,
        fixation,
        rsvp_maker(item_1),
        between_items,
        rsvp_maker(item_2),
        between_items,
        rsvp_maker(item_3),
        between_items,
        rsvp_maker(item_4),
        between_items,
        rsvp_maker(item_5),
        between_items,
        rsvp_maker(item_6),
        between_items,
        rsvp_maker(item_7),
        rsvp_maker(item_8),
        between_items,
        response,
        response,
        fixation_offset,
    ]

    training_block = Block(training_list, training_timeline)

    experiment_boarding_list = [experiment_boarding]
    experiment_boarding_block = Block(experiment_boarding_list)

    experiment_list = [
        fixation_onset,
        fixation,
        rsvp_maker(item_1),
        between_items,
        rsvp_maker(item_2),
        between_items,
        rsvp_maker(item_3),
        between_items,
        rsvp_maker(item_4),
        between_items,
        rsvp_maker(item_5),
        between_items,
        rsvp_maker(item_6),
        between_items,
        rsvp_maker(item_7),
        rsvp_maker(item_8),
        between_items,
        response,
        response,
        fixation_offset,
    ]

    # # test for one item per trial
    # item = TimelineVariable('item', [])
    # rdp = rdp_rsvp_stimulus(
    #     duration = 500,
    #     number_of_oobs=50,
    #     number_of_apertures=1,
    #     movement_speed=20,
    #     coherence_movement=coherence_ratio,
    #     coherent_movement_direction=motion_direction,
    #     coherent_orientation=motion_direction,
    #     oob_color="white",
    #     background_color="black",
    #     # choices   =choices,
    #     stimulus_type=1, # 1 is for circles
    #     text = item,
    #     prompt=item,
    #     color="black"
    #     )
    # experiment_list = [rdp, fixation_offset]
    experiment_block = Block(experiment_list, experiment_timeline)

    # response_list = [response, response]
    # respone_block = Block(response_list)

    debriefing_list = [debriefing]
    debriefing_block = Block(debriefing_list)

    # setting up the final experiment consisting of all blocks
    block_list = [
        introduction_block,
        instruction_block,
        training_boarding_block,
        training_block,
        experiment_boarding_block,
        experiment_block,
        debriefing_block,
    ]
    # block_list = [introduction_block, instruction_block, experiment_block, debriefing_block]

    experiment = Experiment(block_list)

    return experiment.to_html("test_experiment.html")
    # return experiment.to_js_string(as_function=True, is_async=True)
