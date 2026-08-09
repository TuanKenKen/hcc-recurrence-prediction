# HCC Recurrence Prediction

Machine learning models for predicting hepatocellular carcinoma (HCC) recurrence.

## Overview

This repository contains three machine learning approaches for HCC recurrence prediction:

* Logistic Regression
* XGBoost
* Multilayer Perceptron (MLP)

It also contains scripts used to inspect and check the dataset before model training.

## Repository Contents

### Model training

* `train_logistic_regression.py` — trains and evaluates the Logistic Regression model
* `train_xgboost.py` — trains and evaluates the XGBoost model
* `train_mlp.py` — trains and evaluates the Multilayer Perceptron model

### Data checking

* `check_categories.py` — checks categorical variables and their values
* `check_rare_categories_ES_DMH.py` — checks rare categories in the ES and DMH variables
* `check_skewness.py` — examines skewness in numerical variables

## Data

The original clinical dataset used for this project is **not included in this repository** because it contains confidential patient information.

A fully synthetic dataset is provided for demonstration and code-testing purposes.

The synthetic dataset:

* follows the same general structure as the original dataset;
* uses artificial records rather than real patients;
* contains no real patient-level information;
* allows the scripts to be tested without exposing confidential data.

Results obtained using the synthetic dataset should **not be interpreted as evidence of clinical model performance**.

## Running the Code

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/hcc-recurrence-prediction.git
cd hcc-recurrence-prediction
```

Install the required Python packages used by the scripts.

Then run the required model, for example:

```bash
python train_logistic_regression.py
```

or:

```bash
python train_xgboost.py
```

or:

```bash
python train_mlp.py
```

The data-checking scripts can be run in the same way:

```bash
python check_categories.py
python check_rare_categories_ES_DMH.py
python check_skewness.py
```

## Important Note

This repository is intended for research, development, and demonstration purposes.

The synthetic dataset is provided only to demonstrate the workflow and allow the code to run without sharing confidential clinical data. Model performance on synthetic data should not be considered representative of performance on real-world HCC patients.

## Project Status

This project is part of ongoing work on machine learning approaches for predicting recurrence of hepatocellular carcinoma.
