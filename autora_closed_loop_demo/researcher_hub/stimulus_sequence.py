from sweetbean.stimulus import FixationStimulus, BlankStimulus, RandomDotPatternsStimulus
from sweetbean.parameter import TimelineVariable
from sweetbean.sequence import Block, Experiment

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