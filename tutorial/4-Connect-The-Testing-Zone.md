# Testing Zone: Firebase Website

The `testing_zone` contains a basic template for a website that is compatible with the [AutoRA Experimentation Manager for Firebase](https://autoresearch.github.io/autora/user-guide/experiment-runners/experimentation-managers/firebase/) and the [AutoRA Recruitment Manager for Prolific](https://autoresearch.github.io/autora/user-guide/experiment-runners/recruitment-managers/prolific/).

## Copy Web App Credentials

- Navigate to the [Firebase console](https://console.firebase.google.com/) and select the project
- On the gear-symbol next to `Project Overview`, you can find `project settings`. You will find credentials in the tab `general` (you might have to scroll down). Copy the credentials to the corresponding variables in the `.env` file in the `testing_zone` folder that was created on your system using create-react-app or cookiecutter
```dotenv
REACT_APP_apiKey=
REACT_APP_authDomain=
REACT_APP_projectId=
REACT_APP_storageBucket=
REACT_APP_messagingSenderId=
REACT_APP_appId=
REACT_APP_devNoDb="True"
REACT_APP_useProlificId="False"
```

## Configure Your Project For Firebase
In the `testing_zone` folder, enter the following commands in your terminal:
First log in to your Firebase account using
```shell
firebase login
```
or (if you run this in codespace)
```shell
firebase login --no-localhost
```
Then initialize the Firebase project in this folder by running:
```shell
firebase init
```
An interactive initialization process will now run in your command line. For the first question, select these options:

- Firestore: Configure security rules and indexes files for Firestore
- Hosting: Configure files for Firebase Hosting and (optionally) set up GitHub Action deploys
- For a Firebase project, use the one you created earlier
- Use the default options for the Firestore rules and the Firestore indexes.
- ***!!! IMPORTANT !!!*** Use the build directory instead of the public directory here.
- When asked for the directory, write `build` and press `Enter`.
- Configure as a single-page app; don't set up automatic builds and deploys with GitHub. 
- Don't overwrite the index.html file if the question pops up.

## Add jspsych
Next, we install jspsych (in the testing_zone):
```shell
npm install jspsych@7.3.1
```
and minimal package:
```shell
npm install @jspsych/plugin-html-keyboard-response
```


## Build And Deploy To Firebase 
To serve the website on the internet, you must build and deploy it to Firebase.
To build the project, run
```shell
npm run build
```
To deploy to Firebase, run
```shell
firebase deploy
```

***
Next: [5-Setup-AutoRA](./5-Setup-AutoRA.md)
***

**Navigate Through the Tutorial**:
- Use the navigation buttons in the preview pane to move through the tutorial steps.
- After pressing on links to navigate, click on the left editor window to refocus.
