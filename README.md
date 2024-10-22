# Figure Eight

## Table of contents

- [Installation](#installation)
- [Instructions](#instructions)
- [Project motivation](#project-motivation)
- [File descriptions](#file-descriptions)
- [Results](#results)
- [Creator](#creator)
- [Thanks](#thanks)


## Installation

In order to be able to execute your own python statements it should be noted that scripts are only tested on **anaconda distribution 4.5.11** in combination with **python 3.6.6**. The scripts require additional python libraries.

Run the following commands in anaconda prompt to be able to run the scripts that are provided in this git repository.
- `conda install scikit-learn`
- `conda install pandas`
- `conda install numpy`
- `conda install -c conda-forge nltk_data`
- `conda install joblib`
- `conda install plotly`
- `conda install flask`

For this project a SQLite database was used on my local machine. If you want to follow along and execute these scripts you will need to [install SQLite](http://www.sqlitetutorial.net/download-install-sqlite/) on your machine.

Two quick start options are available:
- [Download the latest release.](https://github.com/FrankTub/Figure8/zipball/master/)
- Clone the repo: `git clone https://github.com/FrankTub/Figure8.git`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3. Open your favorite browser and navigate to the following URL:
    `http://localhost:3001`

## Project motivation
For the second term of the nanodegree [become a data scientist](https://eu.udacity.com/course/data-scientist-nanodegree--nd025) of [Udacity](https://eu.udacity.com/) I got involved in this project. I was particular interested in trying new machine learning algorithms to help [Figure Eight](https://www.figure-eight.com/) detect what messages actually need attention during the event of a disaster. It was very interesting to try out the NLTK library and see how much of the coding is already done for you.

## File descriptions

Within the download you'll find the following directories and files.

```text
Figure8/
├── README.md
├── ETL Pipeline Preparation.ipynb # Notebook to prepare cleaning function
├── ML Pipeline Preparation.ipynb # Notebook to test out machine learning algorithms
├── app/
    ├──	run.py # Flask file that runs app
    ├── templates/
    |       ├──	master.html  # main page of web app
    |       └── go.html  # classification result page of web app  
├── data/
    ├── categories.csv  # data to process
    ├──	messages.csv  # data to process
    ├── process_data.py
    └── DisasterResponse.db   # database to save clean data to
└── models/
    ├── train_classifier.py
    └──	classifier.pkl  # saved model, not stored in github repository due to size, run ML pipeline to create this model.
```

## Results
The model does not perform very well. I've tried some optimization techniques to increase the predictability of the model, but this did not led to an increase of correct predictions. One of the reasons this model did not perform well might be that the data is very skewed, some categories are very rare in the provided dataset, therefore making it hard for algorithms to make correct predictions on these rare conditions.

## Creator

**Frank Tubbing**

- <https://github.com/FrankTub>


## Thanks

<a href="https://eu.udacity.com/">
  <img src="https://eu.udacity.com/assets/iridium/images/core/header/udacity-wordmark.svg" alt="Udacity Logo" width="490" height="106">
</a>

Thanks to [Udacity](https://eu.udacity.com/) for setting up the projects where we can learn cool stuff!

<a href="https://www.figure-eight.com/">
  <img src="https://upload.wikimedia.org/wikipedia/en/a/a6/Attached_to_figure-eight-dot-com.png" alt="Figure Eight Logo">
</a>

Thanks to [Figure Eight](https://www.figure-eight.com/) for providing cool data with which we can create a cutting edge project!
