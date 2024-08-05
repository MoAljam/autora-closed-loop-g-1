# Closed Loop Online Experiment

To establish an online closed-loop for AutoRA, there are two key components that need to be configured:

1. AutoRA Workflow
    - This workflow can be executed locally, on a server, or using `Cylc`. It must have the ability to communicate with a website, allowing for the writing of new conditions and reading of observation data.
    - The AutoRA workflow can be customized by adding or removing AutoRA functions, such as AutoRA *experimentalists* or AutoRA *theorists*. It relies on an AutoRA Prolific Firebase *runner* to collect data from an online experiment hosted via Firebase and recruit participants via prolific.

2. Website To Conduct Experiment:
    - The website serves as a platform for conducting experiments and needs to be compatible with the AutoRA workflow.
    - In this setup, we use `Firebase` to host on website.

To simplify the setup process, we provide a `cookiecutter` template that generates a project folder containing the following two directories:

1. Researcher Hub:
    - This directory includes a basic example of an AutoRA workflow.

2. Testing Zone:
    - This directory provides a basic example of a website served with Firebase, ensuring compatibility with the AutoRA workflow.

***
Next: [2-Setup-Local-Project](./2-Setup-Local-Project.md)
***

**Navigate Through the Tutorial**:
- Use the navigation buttons in the preview pane to move through the tutorial steps.
- After pressing on links to navigate, click on the left editor window to refocus.