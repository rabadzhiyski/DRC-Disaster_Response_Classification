# DRC - Disaster_Response_Classification

Analyzing data for disaster response based on messages generated through different channels. The program extracts the messages, transforms, loads, and runs ML based on RandomForestClassifier to classify messages into 35 different categories.

This repository contains code for a web app which could be used during a disaster event (e.g. an earthquake or hurricane), to classify a disaster message into several categories,so that the appropriate aid agencies are notified.


## Libraries

json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle.


## Contents

app

| - template

| |- master.html  (main page of web app)

| |- go.html  (classification result page of web app)

|- run.py  (Flask file that runs app, it takes around 20-30min to finish based on system capabilities)

data

|- disaster_categories.csv  (data to process)

|- disaster_messages.csv  (data to process)

|- process_data.py (ETL pipeline)

|- DisasterResponse.db   (database to save clean data to)

models

|- train_classifier.py (ML pipeline)

|- classifier.pkl  (THIS IS NOT UPLOADED DUE TO SIZE LIMITATIONS (file is 140-160MB). As soon as the ML model runs locally, it will create this pkl file to your machine)

ETL Pipeline Preparation.ipynb (draft working file)

ML Pipeline Preparation.ipynb (draft working file)

README.md
 

## Instructions

1. Download the folders to your local machine
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/


## Warning
The datasets included in this repository are very unbalanced, with just a few positive examples. In such cases, even though the classifier accuracy is high, the classifier recall (i.e. the proportion of positive examples that were correctly labelled) tends to be very low. As a result, care should be taken if relying on the results of this app for decision making purposes.


## Acknowledgements

This app was completed as part of the [Udacity Data Science Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). Code templates and data were provided by Udacity. The data was originally sourced by Udacity from [Figure Eight](https://appen.com/), now known as Appen.
