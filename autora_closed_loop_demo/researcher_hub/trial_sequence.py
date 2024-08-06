from sweetpea import Factor, MinimumTrials, CrossBlock, synthesize_trials, CMSGen, experiments_to_dicts

def trial_sequences(intensity_1, intensity_2, nr_trials):
    s1 = Factor('S1', [intensity_1, intensity_2])
    s2 = Factor('S2', [intensity_2, intensity_1])
    design = [s1, s2]
    crossing = [s1, s2]
    constraints = [MinimumTrials(nr_trials)]

    block = CrossBlock(design, crossing, constraints)

    # synthesize trialsequence
    experiment = synthesize_trials(block, 1, CMSGen)

    # export as dict:
    return experiments_to_dicts(block, experiment)