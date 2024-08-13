import sweetbean

from sweetbean.parameter import TimelineVariable
from sweetbean.sequence import Block, Experiment
from sweetbean.stimulus import TextStimulus, BlankStimulus, FeedbackStimulus, RandomDotPatternsStimulus


def stimulus_sequence(timeline, coherence_ratio, motion_direction):
    # introduction TODO write introduction
    introduction = TextStimulus(text='Welcome to this experiment!<br>...<br>Press SPACE to continue.', choices=[" "])

    # instruction TODO write instruction
    instruction = TextStimulus(text='... Press SPACE to continue.', choices=[" "])

    # break TODO incorporate breaks after specific number of trials
    pause = TextStimulus(text='Feel free to take a short break now<br>Press SPACE when you are ready to continue the experiment.', choices=[" "])

    # debriefing/closure TODO write debriefing
    debriefing = TextStimulus(text='... Thank you for your participation!<br>Press SPACE to end the experiment.', choices=[" "])



    # fixation cross and blank screens around it
    fixation_onset = BlankStimulus(duration=680)
    fixation = TextStimulus(duration=915, text="+")
    fixation_offset = BlankStimulus(duration=400)

    # blank screen in between items within a trial
    between_items = BlankStimulus(duration=45)

    # participant response
    response = TextStimulus(text="Which two numbers did you see during the previously displayed sequence?", choices=[1, 2, 3, 4])

    # timeline variables
    # independent variables
    coherence_ratio = TimelineVariable("coherence_ratio", [])
    motion_direction = TimelineVariable("motion_direction", [])

    rdp = RandomDotPatternsStimulus(
        duration=75,
        number_of_oobs=20,
        number_of_apertures=1,
        #coherence_movement=coherence_ratio,
        #movement_speed=14.2
        oob_color="white",
        background_color="black"
    )

    # create all lists of sequences and individual blocks
    introduction_list = [introduction]
    introduction_block = Block(introduction_list)

    instruction_list = [instruction]
    instruction_block = Block(instruction_list)

    training_list = [fixation_onset, fixation, rdp, fixation_offset]
    training_block = Block(training_list)
    experiment_list = [fixation_onset, fixation, rdp, fixation_offset]
    experiment_block = Block(experiment_list, timeline)

    response_list = [response, response]
    respone_block = Block(response_list)

    debriefing_list = [debriefing]
    debriefing_block = Block(debriefing_list)

    # setting up the final experiment consisting of all blocks
    block_list = [introduction_block, instruction_block, training_block, experiment_block, debriefing_block]
    experiment = Experiment(block_list)

    # return a js string to transfer experiment to autora
    #return experiment.to_js_string(as_function=True, is_async=True)


'''
def stimulus_sequence(timeline, intensity_1, intensity_2):
    fixation = FixationStimulus(800)
    blank_1 = BlankStimulus(400)
    blank_2 = BlankStimulus(1000)
    s1 = TimelineVariable('S1', [40, 70])
    s2 = TimelineVariable('S2', [40, 70])


    rdp = RandomDotPatternsStimulus(
        duration=2000,
        number_of_oobs=[s1, s2],
        number_of_apertures=2,
        choices=["y", "n"],
    )
    event_sequence = [fixation, blank_1, rdp, blank_2]

    block = Block(event_sequence, timeline)

    experiment = Experiment([block])
    # return a js string to transfer to autora
    return experiment.to_js_string(as_function=True, is_async=True)
'''