# titanic

## overview

This repository contains reproducible Python and R pipelines (using Docker) for the Titanic survival prediction task.

## Download the Titanic Dataset

This project does not include the dataset.  
You must download it manually from the official Kaggle competition:
[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

After downloading, you will have several CSV files:
gender_submission.csv
test.csv
train.csv

Only train.csv and test.csv are required for this assignment.

## Project structure
```
titanic/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── python/
│   │   ├── Dockerfile
│   │   └── titanic_pipeline.py
│   │
│   └── r/
│       ├── Dockerfile
│       ├── install_packages.R
│       └── titanic_pipeline.R
│
├── python_submission.csv
└── r_submission.csv
```
## Environment Setup
### Python Environment

Dependencies are listed in requirements.txt:

pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1

### R Environment

R dependencies are installed automatically through the install_packages.R script:

install.packages(c("tidyverse", "caret", "glmnet"), repos='https://cran.rstudio.com/')

## Docker Setup

Both Python and R pipelines are containerized for portability and consistent execution.

### Python Dockerfile

Located at: src/python/Dockerfile

FROM python:3.11-slim
WORKDIR /app
COPY /requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY /src/python /app/src/python
CMD ["python", "src/python/titanic_pipeline.py"]

### R Dockerfile

Located at: src/r/Dockerfile

FROM r-base:4.3.1
WORKDIR /app
COPY /src/r /app/src/r
RUN Rscript src/r/install_packages.R
CMD ["Rscript", "src/r/titanic_pipeline.R"]

## How to Run
### Python Pipeline

Build the Docker image:

docker build -t titanic-py -f src/python/Dockerfile .


Run the container:

docker run --rm -v "$PWD/src/data":/app/src/data titanic-py


Output file:

python_submission.csv

### R Pipeline

Build the Docker image:

docker build -t titanic-r -f src/r/Dockerfile .


Run the container:

docker run --rm -v "$PWD/src/data":/app/src/data titanic-r


Output file:

r_submission.csv



## Feature Engineering (Summary)

Both pipelines engineer additional useful features before modeling:

Feature	Description
FamilySize	Combines SibSp + Parch + 1
CabinFirstLetter	Extracts the first character of Cabin
Title	Extracts passenger titles (Mr, Mrs, Miss, etc.)
TicketGroupSize	Number of passengers sharing the same ticket

Missing values are imputed with median (numerical) or mode (categorical),
and categorical variables are encoded for model training.

## Model Training

Python model: Logistic Regression (scikit-learn)

R model: Generalized Linear Model (caret package)

Both models are fitted using the training set (train.csv)

The test.csv file is used only for prediction, not evaluation

## Output Files
File	Description
python_submission.csv	Predictions from Python model
r_submission.csv	Predictions from R model

## Notes

The test set provided by Kaggle does not include true labels,
so accuracy is not computed for test.csv.

Instead, models are validated internally on the training data.
