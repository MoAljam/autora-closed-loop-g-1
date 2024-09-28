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
from autora.experimentalist.grid import grid_pool
from autora.experimentalist.adaptable import adaptable_sample
from autora.experimentalist.novelty import novelty_score_sample
from autora.experimentalist.confirmation import confirmation_score_sample
from autora.experimentalist.model_disagreement import model_disagreement_score_sample
from autora.experiment_runner.firebase_prolific import firebase_runner
from autora.state import StandardState, on_state, Delta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sweetbean.sequence import Block, Experiment
from sweetbean.stimulus import TextStimulus

from trial_sequence import trial_sequences
from stimulus_sequence import stimulus_sequence
from utils import PolynomialRegressor, update_html_script
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
        coherence_ratios=[100, 50, 0],
        motion_directions=[0, 90, 180, 270],
        num_repetitions=2,
        sequence_type="target",
    )

    training_seq = trial_sequences(
        coherence_ratios=[90, 10],
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
def theorist_on_state(experiment_data, variables):
    ivs = [iv.name for iv in variables.independent_variables]
    dvs = [dv.name for dv in variables.dependent_variables]
    x = experiment_data[ivs]
    y = experiment_data[dvs]

    theorist_polyr = PolynomialRegressor()
    theorist_lr = LinearRegression()
    return Delta(models_lr=[theorist_lr.fit(x, y)], models_polyr=[theorist_polyr.fit(x, y)])


# ** Experimentalist ** #
# Here, we use a random pool and use the wrapper to create a on state function
# Note: The argument num_samples is not a state field. Instead, we will pass it in when calling
# the function


@on_state()
def experimentalist_on_state(variables, num_samples, experimentalist=pool):
    return Delta(conditions=experimentalist(variables, num_samples))


# Again, we need to wrap the runner to use it on the state. Here, we send the raw conditions.
@on_state()
def runner_on_state(conditions, experiment_runner: callable):
    # Here, we convert conditions into sweet bean code to send the complete experiment code
    # directly to the server

    coherence_ratios_list = list(conditions["coherence_ratio"])
    motion_directions_list = list(conditions["motion_direction"])
    conditions_to_send = conditions.copy()

    # global training_seq
    # experiment_timeline = trial_sequences(coherence_ratios_list, motion_directions_list, all_items_in_one_trial=True)[0]
    # js_code = stimulus_sequence(experiment_timeline, training_timeline=training_seq, to_html=False)

    experiment_seq = trial_sequences(
        coherence_ratios=[100, 50, 0],
        motion_directions=[0, 90, 180, 270],
        num_repetitions=2,
        sequence_type="target",
    )

    training_seq = trial_sequences(
        coherence_ratios=[90, 10],
        motion_directions=[45],
        num_repetitions=1,
        sequence_type="training",
    )

    print("len training sequence: ", len(training_seq[0]))
    print("len experiment sequence: ", len(experiment_seq[0]))

    # display(pd.DataFrame(experiment_seq[0]).head())
    # display(pd.DataFrame(training_seq[0]).head())

    js_code = stimulus_sequence(experiment_seq[0], training_seq[0], to_html=False)
    conditions_to_send["experiment_code"] = js_code

    # dev
    data_raw = experiment_runner()  # returns observations for each condition as jsPsych data
    print("## got raw data ##")
    print("data lenght", len(data_raw))
    print("data type", type(data_raw), "type of first element", type(data_raw[0]))
    # print("data_raw[0]", data_raw[0])

    # process the experiment data
    experiment_data = pd.DataFrame()
    _df = trial_list_to_experiment_data(data_raw)
    experiment_data = pd.concat([experiment_data, _df], axis=0)
    print("processed experiment_data (head):")
    display(experiment_data.head())
    return Delta(experiment_data=experiment_data)


def trial_list_to_experiment_data(trial_sequence):
    """
    Parse a trial sequence (from jsPsych) into dependent and independent variables
    independent: coherence_ratio, motion_direction
    dependent: d_prime
    """
    trial_sequence = pd.DataFrame(trial_sequence).fillna(pd.NA)
    display(trial_sequence.head())

    # target cleaned up data:
    # index(actual trial index), coherence_ratio, motion_direction, response, bean_correct_key
    # inference: hit, miss, d_prime
    # final: index(actual trial index), coherence_ratio, motion_direction, d_prime

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
    display(rok_trials)

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
    display(response_trials)

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
    display(trials)

    # get only target trials (type: "target")
    trials = trials[trials["type"] == "target"]

    # group the trails on condition (coherence_ratio, motion_direction)
    trials_grouped = trials.groupby(["coherence_ratio", "motion_direction"]).agg({"hit": "sum", "miss": "sum"})
    # calculate d_prime
    # d_prime: d_prime(hit, miss)
    # where hit and misses are aggregated over all trials with the same conditions
    trials_grouped["d_prime"] = trials_grouped.apply(lambda x: d_prime(x["hit"], x["miss"]), axis=1)
    trials_grouped = trials_grouped.reset_index()
    display(trials_grouped)

    # select only the experiment data columns (drop hit and miss and type)
    trials_grouped = trials_grouped.loc[:, ["coherence_ratio", "motion_direction", "d_prime"]]
    display(trials_grouped)
    return trials_grouped


@on_state()
def grid_pool_on_state(variables):
    return Delta(conditions=grid_pool(variables))


@on_state()
def costume_experimentalist_on_state(
    experiment_data,
    variables,
    models_lr,
    models_polyr,
    all_conditions,
    cycle=None,
    max_cycle=None,
    num_samples=1,
    random_state=None,
):
    # temperature(0-1) to determine the progress of the discovery process
    temperature = cycle / max_cycle
    temperature = np.clip(temperature, 0, 1)

    # get the input relevant to some of the samplers
    # independent and dependent variables for the metadata
    iv = variables.independent_variables
    dv = variables.dependent_variables
    meta_data = VariableCollection(independent_variables=iv, dependent_variables=dv)

    # reference conditions and observations
    ivs = [iv.name for iv in variables.independent_variables]
    dvs = [dv.name for dv in variables.dependent_variables]
    reference_conditions = experiment_data[ivs]
    reference_observations = experiment_data[dvs]

    # remove the conditions that have already been sampled from the conditions pool
    # remove reference conditions from the conditions pool
    if isinstance(all_conditions, pd.DataFrame) and isinstance(reference_conditions, pd.DataFrame):
        conditions_pool = pd.concat([all_conditions, reference_conditions])
        conditions_pool = conditions_pool.drop_duplicates(keep=False)
    else:
        conditions_pool = all_conditions[~all_conditions.isin(reference_conditions)].dropna()

    # NOTE: the sampler is performeing a bit worse when including falsification and confirmation
    #       possiblly due to passing only one model to theses samplers
    #       while the performance is based on 3 models
    samplers = [
        {
            "func": novelty_score_sample,
            "name": "novelty",
            "params": {"reference_conditions": reference_conditions},
        },
        # {
        #     "func": falsification_score_sample,
        #     "name": "falsification",
        #     "params": {
        #         "reference_conditions": reference_conditions,
        #         "reference_observations": reference_observations,
        #         "metadata": meta_data,
        #         "model": models_polyr[-1],
        #     },
        # },
        {
            "func": model_disagreement_score_sample,
            "name": "model_disagreement",
            "params": {
                "models": [models_lr[-1], models_polyr[-1]],
            },
        },
        {
            "func": confirmation_score_sample,
            "name": "confirmation",
            "params": {
                "reference_conditions": reference_conditions,
                "reference_observations": reference_observations,
                "metadata": meta_data,
                "model": models_polyr[-1],
            },
        },
    ]
    # samplers_coords = [0, 1, 3, 4, 6]  # optional
    # samplers_coords = [1, 2, 5]

    adaptable_sampler_sensitivity = 14

    new_conditions = adaptable_sample(
        conditions=conditions_pool,
        reference_conditions=reference_conditions,
        models=models_polyr,  # pass only the polyr models
        samplers=samplers,
        num_samples=num_samples,
        # samplers_coords=samplers_coords,
        sensitivity=adaptable_sampler_sensitivity,
        plot_info=False,
    )

    # new_conditions = progressive_sample(
    #     conditions=all_conditions,
    #     num_samples=num_samples,
    #     # models=[models_lr[-1], models_polyr[-1]],
    #     temprature=temperature,
    #     samplers=samplers,
    #    # samplers_coords=samplers_coords,
    # )

    return Delta(conditions=new_conditions)


from autora.state import State
from dataclasses import dataclass, field
from typing import Optional, List
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class CustomState(State):
    variables: Optional[VariableCollection] = field(default=None, metadata={"delta": "replace"})
    conditions: Optional[pd.DataFrame] = field(default=None, metadata={"delta": "replace", "converter": pd.DataFrame})
    experiment_data: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "extend", "converter": pd.DataFrame}
    )
    models_lr: List[BaseEstimator] = field(
        default_factory=list,
        metadata={"delta": "extend"},
    )
    models_polyr: List[BaseEstimator] = field(
        default_factory=list,
        metadata={"delta": "extend"},
    )


# *** Report the data *** #


def get_validation_MSE(validation_experiment_data, working_state):
    ivs = [iv.name for iv in validation_experiment_data.variables.independent_variables]
    dvs = [dv.name for dv in validation_experiment_data.variables.dependent_variables]
    X = validation_experiment_data.experiment_data[ivs]
    y = validation_experiment_data.experiment_data[dvs]

    MSE_values_lr_ = np.zeros(len(working_state.models_lr))
    MSE_values_polyr = np.zeros(len(working_state.models_polyr))
    for idx, (model_lr, model_polyr) in enumerate(zip(working_state.models_lr, working_state.models_polyr)):
        y_pred_lr = model_lr.predict(X)
        y_pred_polyr = model_polyr.predict(X)

        MSE_lr = ((y - y_pred_lr) ** 2).mean()[0]
        MSE_polyr = ((y - y_pred_polyr) ** 2).mean()[0]

        MSE_values_lr_[idx] = MSE_lr
        MSE_values_polyr[idx] = MSE_polyr

    validation_MSE = pd.DataFrame(
        {
            "model_lr": MSE_values_lr_,
            "model_polyr": MSE_values_polyr,
        }
    )
    return validation_MSE


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
            Variable(name="coherence_ratio", allowed_values=np.linspace(0, 100, 100), value_range=(0, 100)),
            Variable(name="motion_direction", allowed_values=np.linspace(0, 360, 360), value_range=(0, 360)),
        ],
        dependent_variables=[Variable(name="d_prime", value_range=(0, 10000))],
    )

    # *** State *** #
    state = CustomState(
        variables=variables,
    )

    validation_conditions = grid_pool_on_state(state)

    # *** experiment runner *** #
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

    # *** Run the close cycle *** #
    # here we use the theorists and experimentalists defined above in the respective functions
    # theorests: [LinearRegression, PolynomialRegressor]
    # Experimentalist: adaptable_sample (with novelty, model_disagreement, confirmation)

    # Collect 'num_samples' conditions per iteration
    num_samples = 2
    iterations = 3
    for i in range(iterations):
        print(f"## Iteration {i} ##")

        if i == 0:  # in the first iteration, sample randomly from the condition space
            state = experimentalist_on_state(state, num_samples=num_samples, experimentalist=pool)
        else:  # for the rest, use the costume experimentalist
            state = costume_experimentalist_on_state(
                state,
                num_samples=num_samples,
                all_conditions=validation_conditions.conditions,
                cycle=i,
                max_cycle=iterations,
            )
        print("## experimentalist done - ", i)
        state = runner_on_state(state, experiment_runner=experiment_runner)
        print("## runner done - ", i)
        state = theorist_on_state(state)
        print("## theorist done - ", i)

    # *** Validation *** #
    # for now we use all the data from the experiments for validation
    validation_experiment_data = state
    validation_MSE = get_validation_MSE(validation_experiment_data, state)

    display(validation_MSE)
    # plot the validation MSE
    import matplotlib.pyplot as plt

    plt.plot(validation_MSE["model_lr"], label="Linear Regression")
    plt.plot(validation_MSE["model_polyr"], label="Polynomial Regression")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
