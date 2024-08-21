from sweetbean.parameter import TimelineVariable
from sweetbean.sequence import Block
from sweetbean.stimulus import (
    TextStimulus,
    BlankStimulus,
    TextSurveyStimulus,
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

    correct_choice = TimelineVariable("correct_choice", [])
    sequence_type = TimelineVariable("sequence_type", [])
    choices = TimelineVariable("choices", [])
    
    # introduction
    introduction = TextStimulus(text="<p>Welcome to the experiment!<br>Let's look into the instructions of the experiment first.</p>Press SPACE to continue.", choices=[" "])

    # instruction
    instruction_1 = TextStimulus(text="<p>In this experiment, you will see a sequence of <strong>eight items</strong>, which consist of <strong>six letters</strong> out of the whole alphabet and <strong>two digits</strong> out of '1', '2', '3', and '4'.<br>The items will be presented one after another in the middle of your screen.</p>Press SPACE to continue.", choices=[" "])

    instruction_2 = TextStimulus(text="<p>Your task will be to focus on the two digits and remember them. You will then also be asked to report them by pressing the according keys on your keyboard. The order in which they were presented doesn't matter here – just the two digits themselves.</p>Press SPACE to continue.", choices=[" "])

    instruction_3 = TextStimulus(text="<p>There will also be some pattern with moving dots displayed in the background, but you don't have to pay attention to that.<br>Try focusing on the sequence, and in particular, the two digits instead.</p>Press SPACE to continue.", choices = [" "])

    # training onboarding
    training_boarding = TextStimulus(text="<p>Now let's look at some examples.<br>In the following, we have a couple of training rounds for you to get accustomed to the experiment and the task.</p>Press SPACE to start the training.", choices=[" "])

    # experiment onboarding
    experiment_boarding = TextStimulus(
        text="<p>Now that you know how the experiment works and what you have to do, let's start with the actual experiment!<br>It will look exactly like the training you just had. Again, try to focus on the sequence and the two digits in the middle of your sceen. Good luck!</p>Press SPACE to start the experiment.", choices=[" "]
    )

    # break
    pause = TextStimulus(
        text="<p>Feel free to take a short break now!</p>Press SPACE when you are ready to continue the experiment.",
        choices=[" "],
    )

    # debriefing/closure
    debriefing = TextStimulus(
        text="<p>Congratulations, you finished the experiment!<br>Thank you for participating and for sticking until the end! I hope you had at least some fun during all of it.</p>Press SPACE to continue.", choices=[" "]
    )

    feedback = TextSurveyStimulus(
        prompts=["<p>If you have any feedback for us, we would appreciate hearing about it. Did you encounter any issues during the experiment? Do you have any suggestions for improvement? Or do you have any other comments you want to share with us? Let us know here!</p>"]
    )

    closure = TextStimulus(
        text="<p>Thank you once again for your participation, and have a great day!</p>Press SPACE to end the experiment.",
        choices=[" "]
    )

    # fixation cross and blank screens around it
    fixation_onset = BlankStimulus(duration=680)
    fixation = TextStimulus(duration=915, text="+")
    fixation_offset = BlankStimulus(duration=400)

    # blank screen in between items within a trial
    between_items = BlankStimulus(duration=45)

    # participant response
    response_1 = TextStimulus(text="<p>Use your keyboard to enter a digit which you recall seeing out of<br>'1', '2', '3', '4'</p>If you cannot remember any number, press 'x'.", choices=["1", "2", "3", "4", "x"],correct_key=correct_choice)
    response_2 = TextStimulus(text="<p>Use your keyboard to enter the other digit you recall seeing out of<br>'1', '2', '3', '4'.</p>If you cannot remember another number, press 'x'.", choices=["1", "2", "3", "4", "x" ], correct_key=correct_choice)

    # timeline variables
    # independent variables

    def rsvp_maker(
        item, coherence_ratio=coherence_ratio, motion_direction=motion_direction, correct_choice=correct_choice, sequence_type=sequence_type
    ):
        rdp = rdp_rsvp_stimulus(
            duration=75,
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
            correct_key=correct_choice
        )
        return rdp

    # create all lists of sequences and individual blocks
    introduction_list = [introduction]
    introduction_block = Block(introduction_list)

    instruction_list = [instruction_1, instruction_2, instruction_3]
    instruction_block = Block(instruction_list)

    training_boarding_list = [training_boarding]
    training_boarding_block = Block(training_boarding_list)

    training_list = [
        fixation_onset,
        fixation,
        fixation_offset,
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
        between_items,
        rsvp_maker(item_8),
        response_1,
        response_2
    ]

    training_block = Block(training_list, training_timeline)

    experiment_boarding_list = [experiment_boarding]
    experiment_boarding_block = Block(experiment_boarding_list)

    experiment_list = [
        fixation_onset,
        fixation,
        fixation_offset,
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
        between_items,
        rsvp_maker(item_8),
        response_1,
        response_2
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

    debriefing_list = [debriefing, feedback, closure]
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
