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

# hard coded trainning sequence
training_seq = trial_sequences(coherence_ratios=[15, 85], motion_directions=[45], all_items_in_one_trial=True)

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

# To use the theorist on the state object, we wrap it with the on_state functionality and return a
# Delta object.
# Note: The if the input arguments of the theorist_on_state function are state-fields like
# experiment_data, variables, ... , then using this function on a state object will automatically
# use those state fields.
# The output of these functions is always a Delta object. The keyword argument in this case, tells
# the state object witch field to update.


@on_state()
def theorist_on_state(experiment_data, variables):
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
def experimentalist_on_state(variables, num_samples):
    return Delta(conditions=pool(variables, num_samples))


# ** Experiment Runner ** #
# We will run our experiment on firebase and need credentials. You will find them here:
# (https://console.firebase.google.com/)
#   -> project -> project settings -> service accounts -> generate new private key

firebase_credentials = {}

# simple experiment runner that runs the experiment on firebase
experiment_runner = firebase_runner(firebase_credentials=firebase_credentials, time_out=100, sleep_time=5)


# Again, we need to wrap the runner to use it on the state. Here, we send the raw conditions.
@on_state()
def runner_on_state(conditions):
    # Here, we convert conditions into sweet bean code to send the complete experiment code
    # directly to the server

    global training_seq
    coherence_ratios_list = list(conditions["coherence_ratio"])
    motion_directions_list = list(conditions["motion_direction"])
    timeline = trial_sequences(coherence_ratios_list, motion_directions_list, all_items_in_one_trial=True)[0]
    js_code = stimulus_sequence(timeline, training_timeline=training_seq)
    conditions_to_send = conditions.copy()
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
    data_raw = experiment_runner(conditions_to_send)  # returns observations for each condition as jsPsych data

    # process the experiment data
    experiment_data = pd.DataFrame()
    for item in data_raw:
        _lst = json.loads(item)["trials"]
        _df = trial_list_to_experiment_data(_lst)  # list of dicts
        experiment_data = pd.concat([experiment_data, _df], axis=0)
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
    res_dict = {"coherence_ratio": [], "motion_direction": [], "d_prime": []}
    for trial in trial_sequence:
        # Filter trials that are not ROK (instructions, fixation, ...)
        if trial["trial_type"] != "rok":
            continue
        # Filter trials without rt
        if "d_prime" not in trial or trial["d_prime"] is None:
            continue
        # the intensity is equivalent to the number of oobs (set in sweetBean script)
        # rt is a default value of every trial
        s1 = trial["number_of_oobs"][0]  # first value in list
        s2 = trial["number_of_oobs"][1]
        rt = trial["d_prime"]
        # key = trial['key_press'] oder trial['correct']

        res_dict["coherence_ratio"].append(int(s1))
        res_dict["motion_direction"].append(int(s2))
        res_dict["d_prime"].append(float(rt))

    dataframe_raw = pd.DataFrame(res_dict)

    # Calculate the mean rt for each S1/S2 combination
    # easiest: add one condition per participants, but this is not always possible
    grouped = dataframe_raw.groupby(["coherence_ratio", "motion_direction"]).mean().reset_index()

    return grouped


# Now, we can run our components
# this is the cycle!
for _ in range(3):
    state = experimentalist_on_state(state, num_samples=2)  # Collect 2 conditions per iteration
    state = runner_on_state(state)
    state = theorist_on_state(state)


# *** Report the data *** #
# If you changed the theorist, also change this part
def report_linear_fit(m: LinearRegression, precision=4):
    s = f"y = {np.round(m.coef_[0].item(), precision)} x " f"+ {np.round(m.intercept_.item(), 4)}"
    return s


print(report_linear_fit(state.models[0]))
print(report_linear_fit(state.models[-1]))
