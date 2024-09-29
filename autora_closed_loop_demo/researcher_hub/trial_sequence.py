from sweetpea import (
    Factor,
    MinimumTrials,
    CrossBlock,
    synthesize_trials,
    CMSGen,
    experiments_to_dicts,
    tabulate_experiments,
)
from sweetpea import MultiCrossBlock
from sweetpea import ExactlyKInARow
from sweetpea import DerivedLevel
import random

import pandas as pd
import numpy as np


def generate_trial_items():
    letters = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    numbers = ["1", "2", "3", "4"]
    # make a 2 lists of 4 items where each list contains 3 letters and 1 number
    items_first = random.sample(letters, 3) + random.sample(numbers, 1)
    items_second = random.sample(letters, 3) + random.sample(numbers, 1)
    # shuffle the items
    random.shuffle(items_first)
    random.shuffle(items_second)
    items = items_first + items_second
    # make a dict with the items from item_1 to item_8
    # trial_items = {f'item_{i}': items[i-1] for i in range(1, 9)}
    # return trial_items
    return items
    # alternative: construct it with sweetpea


def shuffle_chunks(group):
    # Shuffle the group to randomize data before chunking
    group = group.sample(frac=1).reset_index(drop=True)
    n = 8  # Define chunk size
    chunks = [group.iloc[i : i + n] for i in range(0, len(group), n)]

    return chunks


def assign_items_shuffle_chunk(group):
    # Shuffle the group to randomize data before chunking
    group = group.sample(frac=1).reset_index(drop=True)
    items = generate_trial_items()
    if len(group) != len(items):
        raise ValueError("Group and items must have the same length")
    # add one item to each trial from the generated items
    for idx, _ in group.iterrows():
        group.at[idx, "item"] = items[idx]

    n = len(group)
    chunks = [group.iloc[i : i + n] for i in range(0, len(group), n)]
    return chunks


def trial_sequences(
    coherence_ratios: list,
    motion_directions: list,
    sequence_type="target",
    all_items_in_one_trial=True,
    num_repetitions=8,
):

    coherence_ratio = Factor("coherence_ratio", coherence_ratios)
    motion_direction = Factor("motion_direction", motion_directions)
    repetition = Factor("repetetion", list(range(1, num_repetitions + 1)))

    # used in case of repeating the items of each trial
    _num = Factor("_num", list(range(1, num_repetitions + 1)))

    # design = [coherence_ratio, motion_direction, repetetion, *items]
    design = [coherence_ratio, motion_direction, repetition]
    crossing = [coherence_ratio, motion_direction, repetition]

    if not all_items_in_one_trial:
        design.append(_num)
        crossing.append(_num)

    constraints = []

    # block = CrossBlock(design, crossing, constraints)
    block = CrossBlock(design, crossing, constraints)

    # synthesize trialsequence
    experiments = synthesize_trials(block, 1, CMSGen)

    experiments_dicts = experiments_to_dicts(block, experiments)

    # display(experiments_dicts[0])
    if all_items_in_one_trial:
        # extend the the experiment with the trial items
        for experiment in experiments_dicts:
            for trial in experiment:
                trial_items = generate_trial_items()
                items_dict = {f"item_{i}": trial_items[i - 1] for i in range(1, 9)}
                trial.update(items_dict)
                # add the correct response to each trial
                numbers = [int(item) for item in trial_items if item.isdigit()]
                # trial["correct_choice"] = [chr(n) for n in numbers]
                trial["correct_choice"] = numbers
                # add the trail type
                trial["sequence_type"] = sequence_type

    else:
        raise NotImplementedError(
            "Not implemented , use all_items_in_one_trial=True as it fits the current implementation"
        )
        # # convert to pandas dataframe for more convenient manipulation
        # dfs = [pd.DataFrame(experiment) for experiment in experiments_dicts]
        # extended_dfs = []
        # for df in dfs:
        #     extended_chunks = df.groupby(["coherence_ratio", "motion_direction"], group_keys=False).apply(
        #         assign_items_shuffle_chunk
        #     )
        #     extended_chunks = extended_chunks.tolist()
        #     # shuffle the chunks (each chunk is the 8 trials of a coherence_ratio and motion_direction pair)
        #     np.random.shuffle(extended_chunks)
        #     # put the chunks back together
        #     shuffled_extended_df = pd.concat(
        #         [chunk for chunk_list in extended_chunks for chunk in chunk_list], ignore_index=True
        #     )
        #     extended_dfs.append(shuffled_extended_df)
        # # convert back to lists of dicts
        # experiments_dicts = [df.to_dict("records") for df in extended_dfs]

    return experiments_dicts


if __name__ == "__main__":
    trial_seq = trial_sequences(coherence_ratios=[0, 100], motion_directions=[0, 90], all_items_in_one_trial=False)
    print(len(trial_seq[0]))
    # convert to pandas dataframe for better visualization
    df = pd.DataFrame(trial_seq[0])
    print("## as a dataframe: original is a list of dicts")
    print(df)
