"""
Basic Workflow
    Single condition Variable (0-1), Single Observation Variable(0-1)
    Theorist: LinearRegression
    Experimentalist: Random Sampling
    Runner: Firebase Runner (no prolific recruitment)
"""

import json

from autora.variable import VariableCollection, Variable
from autora.experimentalist.random import pool
from autora.experiment_runner.firebase_prolific import firebase_runner
from autora.state import StandardState, on_state, Delta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sweetbean.sequence import Block, Experiment
from sweetbean.stimulus import TextStimulus

from trial_sequence import trial_sequences
from stimulus_sequence import stimulus_sequence
from utils import update_html_script
from methods import d_prime
from IPython.display import display
import os


def psudo_experiment_runner():
    # load a csv file called experiment_data.csv
    if not os.path.exists("myexperiment.csv"):
        raise FileNotFoundError("myexperiment.csv not found")

    raw_data = pd.read_csv("myexperiment.csv")
    return raw_data.to_dict(orient="records")


def run_experiment_once():
    experiment_seq = trial_sequences(
        coherence_ratios=[0, 20, 100],
        motion_directions=[0, 90, 180, 270],
        num_repetitions=5,
        sequence_type="experiment",
    )

    training_seq = trial_sequences(
        coherence_ratios=[10, 90],
        motion_directions=[45],
        num_repetitions=1,
        sequence_type="training",
    )

    experiment_seq = trial_sequences(
        coherence_ratios=[100],
        motion_directions=[0],
        num_repetitions=2,
        sequence_type="experiment",
    )

    training_seq = trial_sequences(
        coherence_ratios=[90],
        motion_directions=[45],
        num_repetitions=1,
        sequence_type="training",
    )

    print("len training sequence: ", len(training_seq[0]))
    print("len experiment sequence: ", len(experiment_seq[0]))

    display(experiment_seq[0])
    display(training_seq[0])

    stimulus_seq = stimulus_sequence(experiment_seq[0], training_seq[0])
    # print(stimulus_seq)
    update_html_script("test_experiment.html")


# To use the theorist on the state object, we wrap it with the on_state functionality and return a
# Delta object.
# Note: The if the input arguments of the theorist_on_state function are state-fields like
# experiment_data, variables, ... , then using this function on a state object will automatically
# use those state fields.
# The output of these functions is always a Delta object. The keyword argument in this case, tells
# the state object witch field to update.


@on_state()
def theorist_on_state(experiment_data, variables, theorist):
    ivs = [iv.name for iv in variables.independent_variables]
    dvs = [dv.name for dv in variables.dependent_variables]
    x = experiment_data[ivs]
    y = experiment_data[dvs]
    return Delta(models=[theorist.fit(x, y)])


# ** Experimentalist ** #
# Here, we use a random pool and use the wrapper to create a on state function
# Note: The argument num_samples is not a state field. Instead, we will pass it in when calling
# the function


@on_state()
def experimentalist_on_state(variables, num_samples, experimentalist=pool):
    return Delta(conditions=experimentalist(variables, num_samples))


# Again, we need to wrap the runner to use it on the state. Here, we send the raw conditions.
@on_state()
def runner_on_state(conditions):
    # Here, we convert conditions into sweet bean code to send the complete experiment code
    # directly to the server

    coherence_ratios_list = list(conditions["coherence_ratio"])
    motion_directions_list = list(conditions["motion_direction"])
    conditions_to_send = conditions.copy()

    # global training_seq
    # experiment_timeline = trial_sequences(coherence_ratios_list, motion_directions_list, all_items_in_one_trial=True)[0]
    # js_code = stimulus_sequence(experiment_timeline, training_timeline=training_seq)

    experiment_seq = trial_sequences(
        coherence_ratios=[0, 20, 100],
        motion_directions=[0, 90, 180, 270],
        num_repetitions=5,
        sequence_type="experiment",
    )

    training_seq = trial_sequences(
        coherence_ratios=[10, 90],
        motion_directions=[45],
        num_repetitions=1,
        sequence_type="training",
    )

    experiment_seq = trial_sequences(
        coherence_ratios=[100],
        motion_directions=[0],
        num_repetitions=2,
        sequence_type="experiment",
    )

    training_seq = trial_sequences(
        coherence_ratios=[90],
        motion_directions=[45],
        num_repetitions=1,
        sequence_type="training",
    )

    print("len training sequence: ", len(training_seq[0]))
    print("len experiment sequence: ", len(experiment_seq[0]))

    display(pd.DataFrame(experiment_seq[0]).head())
    display(pd.DataFrame(training_seq[0]).head())

    js_code = stimulus_sequence(experiment_seq[0], training_seq[0])
    conditions_to_send["experiment_code"] = js_code

    # res = []
    # for idx, c in conditions.iterrows():
    #     i_1 = c["coherence_ratio"]
    #     i_2 = c["motion_direction"]
    #     # get a timeline via sweetPea
    #     # can also do different timelines
    #     timeline = trial_sequences([i_1], [i_2], all_items_in_one_trial=True)[0]
    #     # get js code via sweetBeaan
    #     js_code = stimulus_sequence(timeline, training_timeline=training_seq)
    #     res.append(js_code)

    # conditions_to_send = conditions.copy()
    # conditions_to_send["experiment_code"] = res
    # upload and run the experiment, sending is to firestore database as conditions
    # data_raw = experiment_runner(conditions_to_send)  # returns observations for each condition as jsPsych data
    # dev
    data_raw = experiment_runner()  # returns observations for each condition as jsPsych data
    print("## got raw data ##")
    print("data lenght", len(data_raw))
    print("data type", type(data_raw), "type of first element", type(data_raw[0]))
    # print("data_raw[0]", data_raw[0])

    # process the experiment data
    experiment_data = pd.DataFrame()
    # for item in data_raw:
    #     _lst = json.loads(item)["trials"]
    #     _df = trial_list_to_experiment_data(_lst)  # list of dicts
    #     experiment_data = pd.concat([experiment_data, _df], axis=0)

    _df = trial_list_to_experiment_data(data_raw)
    experiment_data = pd.concat([experiment_data, _df], axis=0)
    print("processed experiment_data:")
    display(experiment_data)
    return Delta(experiment_data=experiment_data)


"""
    numbers = list(conditions['number'])
    res = []
    for number in numbers:
        # we use sweetbean to create a full experiment from the numbers

        # For more information on sweetbean: https://autoresearch.github.io/sweetbean/
        text = TextStimulus(
            duration=10000, text=f"press a if {number} is larger then 20, b if not.",
            color="purple", choices=["a", "b"]
        )
        block = Block([text])
        experiment = Experiment([block])
        # here we export the experiment as javascript function
        condition = experiment.to_js_string(as_function=True, is_async=True)
        res.append(condition)
    # We append a column (experiment_code) to the conditions and send it to the runner
    conditions_to_send = conditions.copy()
    conditions_to_send['experiment_code'] = res
    # Here, parse the return value of the runner. The return value depends on the specific
    # implementation of your online experiment (see testing_zone/src/design/main.js).
    # In this example, the experiment runner returns a list of strings, that contain json formatted
    # dictionaries.
    # Example:
    # data = ['{'number':4, rt':.8}', ...]
    result = []
    for item in data:
        result.append(json.loads(item))
    return Delta(experiment_data=pd.DataFrame(result))
"""


def trial_list_to_experiment_data(trial_sequence):
    """
    Parse a trial sequence (from jsPsych) into dependent and independent variables
    independent: coherence_ratio, motion_direction
    dependent: d_prime
    """
    trial_sequence = pd.read_csv("../../myexperiment.csv").fillna(pd.NA)
    
    trial_sequence["trial_number"] = np.nan
    index = 1
    counter = 1
    for i in range(26, len(trial_sequence) - 2):
        if counter > 20:
            counter = 1
            index = index +1
        trial_sequence.loc[i, "trial_number"] = index
        counter = counter + 1
        print(counter, "counter")
        
    # print(trial_sequence.loc[41, "bean_choices"])
    # print(trial_sequence.isnull())
    # exit()
    # trial_sequence.replace(r'^\s*$', pd.NA, regex=True)
    trial_sequence =  trial_sequence.ffill() # translate coherent_ration and movement_direction to the input row. 
    trial_sequence.to_csv("vi.csv") 
    trial_sequence= trial_sequence[(trial_sequence['bean_choices'] == '["1","2","3","4","x"]') & (trial_sequence["trial_number"] > 0)] # remove all other rows
    trial_sequence = trial_sequence[['coherence_movement','coherent_movement_direction', 'trial_number', 'response', 'bean_correct_key']] # remove unnecessary columns
    trial_sequence = trial_sequence.groupby(['trial_number', 'bean_correct_key', 'coherence_movement', 'coherent_movement_direction']).agg(lambda x: tuple(x)) 
    trial_sequence['hit'] = pd.NA
    trial_sequence['miss'] = pd.NA
    for trial in trial_sequence: 
        if sorted(trial['response']) == sorted(trial['bean_correct_key']):
            trial['hit'] = 2
        elif trial["response"].isin(trial['bean_correct_key']):
            trial['hit'] = 1
            trial['miss'] = 1
        else:
            trial['miss'] = 2

    d_prime(trial_sequence['hit'].sum , trial_sequence['miss'].sum)
    trial_sequence.to_csv("vi_2.csv") 
    exit()
    
    exit()
    # exit()
    
    
        
        

    # return surogate data
    # data = {
    #     "coherence_ratio": [0, 20],
    #     "motion_direction": [0, 0],
    #     "d_prime": [0.5, 0.6],
    # }
    # return pd.DataFrame(data)

    res_dict = {"coherence_ratio": [], "motion_direction": [], "d_prime": []}
    for trial in trial_sequence:
        # Filter trials that are not ROK (instructions, fixation, ...)
        if trial["sequence_type"] != "experiment":

            continue
        # Filter trials without rt
        if "d_prime" not in trial or trial["d_prime"] is None:
            continue
        # the intensity is equivalent to the number of oobs (set in sweetBean script)
        # rt is a default value of every trial
        coherence_ration = trial["number_of_oobs"][0]  # first value in list
        motion_direction = trial["number_of_oobs"][1]
        d_prime = trial["d_prime"]
        # key = trial['key_press'] oder trial['correct']

        res_dict["coherence_ratio"].append(int(coherence_ration))
        res_dict["motion_direction"].append(int(motion_direction))
        res_dict["d_prime"].append(float(d_prime))

    dataframe_raw = pd.DataFrame(res_dict)

    # Calculate the mean rt for each S1/S2 combination
    # easiest: add one condition per participants, but this is not always possible
    grouped = dataframe_raw.groupby(["coherence_ratio", "motion_direction"]).mean().reset_index()

    return grouped


# *** Report the data *** #
# If you changed the theorist, also change this part
def report_linear_fit(m: LinearRegression, precision=4):
    s = f"y = {np.round(m.coef_[0].item(), precision)} x " f"+ {np.round(m.intercept_.item(), 4)}"
    return s


if __name__ == "__main__":

    run_experiment_once()
    # to run the experiment once localy and download the data
    # -> either uncomment the above line
    # OR
    # -> run it from terminal
    # python
    # >>> from autora_workflow import run_experiment_once
    # >>> run_experiment_once()

    # *** Set up variables *** #
    # independent variable is coherence in percent (0 - 100)
    # dependent variable is rt in ms (0 - 10000)
    variables = VariableCollection(
        independent_variables=[
            Variable(name="coherence_ratio", allowed_values=np.linspace(0, 100, 100)),
            Variable(name="motion_direction", allowed_values=np.linspace(0, 360, 360)),
        ],
        dependent_variables=[Variable(name="d_prime", value_range=(0, 10000))],
    )

    # *** State *** #
    # With the variables, we can set up a state. The state object represents the state of our
    # closed loop experiment.

    state = StandardState(
        variables=variables,
    )

    # *** Components/Agents *** #
    # Components are functions that run on the state. The main components are:
    # - theorist
    # - experiment-runner
    # - experimentalist
    # See more about components here: https://autoresearch.github.io/autora/

    # ** Theorist ** #
    # Here we use a linear regression as theorist, but you can use other theorists included in
    # autora (for a list: https://autoresearch.github.io/autora/theorist/)

    theorist = LinearRegression()

    # ** Experiment Runner ** #
    # We will run our experiment on firebase and need credentials. You will find them here:
    # (https://console.firebase.google.com/)
    #   -> project -> project settings -> service accounts -> generate new private key

    firebase_credentials = {}

    # simple experiment runner that runs the experiment on firebase
    # experiment_runner = firebase_runner(firebase_credentials=firebase_credentials, time_out=100, sleep_time=5)
    # DEV
    experiment_runner = psudo_experiment_runner

    # Now, we can run our components
    # this is the cycle!
    print("## Start the cycle ##")
    for _ in range(3):
        print(f"## Iteration {_} ##")
        state = experimentalist_on_state(
            state, num_samples=2, experimentalist=pool
        )  # Collect 2 conditions per iteration
        print("## experimentalist done")
        state = runner_on_state(state)
        print("## runner done")
        state = theorist_on_state(state, theorist=theorist)
        print("## theorist done")

    # print(report_linear_fit(state.models[0]))
    # print(report_linear_fit(state.models[-1]))
