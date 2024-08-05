# Sweet, sweet Runner

Now that we have the sweetPea and sweetBean function, we can use them in the experiment runner in `aurora_workflow.py`. 

## Update Website

First, we add dependencies to our website:

head over to `testing_zone` and install the jspsych dependency:

```shell
npm install @jspsych-contrib/plugin-rok
```

Also make sure to include the following lines in the `main.js` file in `testing_zone/src/design`:
```javascript
import jsPsychRok from '@jspsych-contrib/plugin-rok'
global.jsPsychRok = jsPsychRok
```

From now on, we also will use the full and do no processing in javascript: 
```javascript
const main = async (id, condition) => {
    const observation = await eval(condition['experiment_code'] + "\nrunExperiment();");
    return JSON.stringify(observation)
}
```

once this is done, rebuild and redeploy your websits:

```shell
npm run build
firebase deploy
```

## Update AutoRA workflow:

Let's import the functions:

```python
from trial_sequence import trial_sequences
from stimulus_sequence import stimulus_sequence
```

First, update the variables:

```python
variables = VariableCollection(
    independent_variables=[
        Variable(name="S1", allowed_values=np.linspace(1, 100, 100)),
        Variable(name="S2", allowed_values=np.linspace(1, 100, 100)),
        ],
    dependent_variables=[Variable(name="rt", value_range=(0, 10000))])
```

Then, we update the experiment runner:

```python
@on_state()
def runner_on_state(conditions):
    res = []
    for idx, c in conditions.iterrows():
        i_1 = c['S1']
        i_2 = c['S2']
        # get a timeline via sweetPea
        timeline = trial_sequences(i_1, i_2, 10)[0]
        # get js code via sweetBeaan
        js_code = stimulus_sequence(timeline, i_1, i_2)
        res.append(js_code)
    
    conditions_to_send = conditions.copy()
    conditions_to_send['experiment_code'] = res
    # upload and run the experiment:
    data_raw = experiment_runner(conditions_to_send)

    # process the experiment data
    experiment_data = pd.DataFrame()
    for item in data_raw:
        _lst = json.loads(item)['trials']
        _df = trial_list_to_experiment_data(_lst)
        experiment_data = pd.concat([experiment_data, _df], axis=0)
    return Delta(experiment_data=experiment_data)
```

Now, all that is left is to implement the function "trial_list_to_experiment". Here, we average rt over intensity_1 and intensity_2 combinations and filter:

```python
def trial_list_to_experiment_data(trial_sequence):
    """
    Parse a trial sequence (from jsPsych) into dependent and independent variables
    independent: S1, S2
    dependent: rt
    """
    res_dict = {
        'S1': [],
        'S2': [],
        'rt': []
    }
    for trial in trial_sequence:
        # Filter trials that are not ROK (instructions, fixation, ...)
        if trial['trial_type'] != 'rok':
            continue
        # Filter trials without rt
        if 'rt' not in trial or trial['rt'] is None:
            continue
        # the intensity is equivalent to the number of oobs (set in sweetBean script)
        # rt is a default value of every trial
        s1 = trial['number_of_oobs'][0]
        s2 = trial['number_of_oobs'][1]
        rt = trial['rt']
        
        res_dict['S1'].append(int(s1))
        res_dict['S2'].append(int(s2))
        res_dict['rt'].append(float(rt))
    
    dataframe_raw = pd.DataFrame(res_dict)
    
    # Calculate the mean rt for each S1/S2 combination
    grouped = dataframe_raw.groupby(['S1', 'S2']).mean().reset_index()

    return grouped
```

***
Next: [9-Add-Feedback](./9-Add-Feedback.md)
***

**Navigate Through the Tutorial**:
- Use the navigation buttons in the preview pane to move through the tutorial steps.
- After pressing on links to navigate, click on the left editor window to refocus.

