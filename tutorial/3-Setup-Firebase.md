# Set Up The Project On The Firebase Website

To serve a website via Firebase and use the Firestore Database, it is necessary to set up a Firebase project. Follow the steps below to get started:

## Google Account
You'll need a [Google account](https://www.google.com/account/about/) to use Firebase.

## Firebase Project
While logged in into your Google account, head over to the [Firebase website](https://firebase.google.com/). Then, create a new project:

- Click on `Get started`.
- Click on the plus sign with `add project`.
- Name your project and click on `continue`.
- For now, we don't use Google Analytics (you can leave it enabled if you want to use it in the future).
- Click `Create project`.

## Adding A Webapp To Your Project
Now, we add a webapp to the project. Navigate to the project and follow these steps:

- Click on ```<\>```.
- Name the app (can be the same as your project) and check the box `Also set up Firebase Hosting`. Click on `Register app`.
- We will use `npm`. We will use the configuration details later, but for now, click on `Next`.
- We will install firebase tools later, for now, click on `Next`.
- We will login and deploy our website later, for now, click on `Continue to console`.

## Adding Firestore To Your Project
For the online closed loop system, we will use a Firestore Database to communicate between the AutoRA workflow and the website conducting the experiment. We will upload experiment conditions to the database and store experiment data in the database. To build a Firestore Database, follow these steps:

- In the left-hand menu of your project console, click on `Build` and select `Firestore Database`.
- Click on `Create database`.
- Select a Firestore location and click on `Next`.
- Select `Start in production mode` selected and click on `Next`.

***
Next: [4-Connect-The-Testing-Zone](./4-Connect-The-Testing-Zone.md)
***

**Navigate Through the Tutorial**:
- Use the navigation buttons in the preview pane to move through the tutorial steps.
- After pressing on links to navigate, click on the left editor window to refocus.