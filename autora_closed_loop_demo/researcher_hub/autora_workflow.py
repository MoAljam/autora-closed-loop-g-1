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

    js_code = stimulus_sequence(experiment_seq[0], training_seq[0], to_html=True)
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
    # js_code = stimulus_sequence(experiment_timeline, training_timeline=training_seq, to_html=False)

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

    js_code = stimulus_sequence(experiment_seq[0], training_seq[0], to_html=False)
    conditions_to_send["experiment_code"] = js_code

    # res = []
    # for idx, c in conditions.iterrows():
    #     i_1 = c["coherence_ratio"]
    #     i_2 = c["motion_direction"]
    #     # get a timeline via sweetPea
    #     # can also do different timelines
    #     timeline = trial_sequences([i_1], [i_2], all_items_in_one_trial=True)[0]
    #     # get js code via sweetBeaan
    #     js_code = stimulus_sequence(timeline, training_timeline=training_seq, to_html=True)
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
    trial_sequence = pd.DataFrame(trial_sequence).fillna(pd.NA)
    # display(trial_sequence.head())

    # target cleaned up data:
    # index(actual trial index), coherence_ratio, motion_direction, response, bean_correct_key
    # inference: hit, miss, d_prime
    # final: index(actual trial index), coherence_ratio, motion_direction, d_prime

    trial_sequence["trial_number"] = np.nan
    trial_sequence["hit"] = np.nan
    trial_sequence["miss"] = np.nan
    trial_sequence["d_prime"] = np.nan

    # get all rok trials
    rok_trials = trial_sequence[trial_sequence["trial_type"] == "rok"]
    # get only relevant columns
    rok_trials = rok_trials.loc[:, ["bean_text", "coherence_movement", "coherent_movement_direction"]]
    rok_trials = rok_trials.rename(
        columns={
            "bean_text": "type",
            "coherence_movement": "coherence_ratio",
            "coherent_movement_direction": "motion_direction",
        }
    )
    # get rid of duplicated information (each 8 rows are one actual trial)
    rok_trials = rok_trials.reset_index(drop=True)
    rok_trials = rok_trials[::8]
    rok_trials = rok_trials.reset_index(drop=True)
    # display(rok_trials)
    # get all responses trails
    # all the html-keyboard-response where bean_correct_key in not null or empty or NA / NaN
    response_trials = trial_sequence[
        (trial_sequence["trial_type"] == "html-keyboard-response") & (trial_sequence["bean_correct_key"].notna())
    ]

    # get only relevant columns
    response_trials = response_trials.loc[:, ["response", "bean_correct_key"]]
    response_trials = response_trials.rename(columns={"bean_correct_key": "correct_response"})
    # put each 2 response trials after each others into one row
    # pair the responses
    responses = response_trials["response"].values.reshape(-1, 2).tolist()
    # make sure responses are floats
    responses = [[float(r) for r in response] for response in responses]
    # get rid of the duplicates
    response_trials = response_trials[::2]
    response_trials["response"] = responses
    # convert correct_response from a string list "[1,2]" to a list [1,2]
    response_trials["correct_response"] = response_trials["correct_response"].apply(lambda x: json.loads(x))
    response_trials = response_trials.reset_index(drop=True)
    # display(response_trials)

    # merge the two dataframes
    trials = pd.concat([rok_trials, response_trials], axis=1)

    # infer the hit and miss per trial
    # hit: number of elements in the
    # miss: number of elements in the correct_response that are not in the response
    def num_hits(array_1, array_2):
        sorted_array_1 = np.sort(array_1)
        sorted_array_2 = np.sort(array_2)
        return np.sum(sorted_array_1 == sorted_array_2)

    def num_misses(array_1, array_2):
        sorted_array_1 = np.sort(array_1)
        sorted_array_2 = np.sort(array_2)
        return np.sum(sorted_array_1 != sorted_array_2)

    trials["hit"] = trials.apply(lambda x: num_hits(x["response"], x["correct_response"]), axis=1)
    trials["miss"] = trials.apply(lambda x: num_misses(x["response"], x["correct_response"]), axis=1)
    # display(trials)

    # group the trails on condition (coherence_ratio, motion_direction)
    trials_grouped = trials.groupby(["coherence_ratio", "motion_direction"]).agg({"hit": "sum", "miss": "sum"})
    # calculate d_prime
    # d_prime: d_prime(hit, miss)
    # where hit and misses are aggregated over all trials with the same conditions
    trials_grouped["d_prime"] = trials_grouped.apply(lambda x: d_prime(x["hit"], x["miss"]), axis=1)
    trials_grouped = trials_grouped.reset_index()
    # select only the experiment data columns (drop hit and miss)
    trials_grouped = trials_grouped.loc[:, ["coherence_ratio", "motion_direction", "d_prime"]]
    # display(trials_grouped)
    return trials_grouped


# *** Report the data *** #
# If you changed the theorist, also change this part
# def report_linear_fit(m: LinearRegression, precision=4):
#     s = f"y = {np.round(m.coef_[0].item(), precision)} x " f"+ {np.round(m.intercept_.item(), 4)}"
#     return s


if __name__ == "__main__":

    # run_experiment_once()
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
