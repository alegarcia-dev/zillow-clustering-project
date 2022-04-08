# What is driving the errors in Zestimates?

This repository contains all deliverables for the Zillow clustering project including additional files used 
in the process of producing the final deliverables.

**Repository Format**
- README.md: Contains a full outline of the project as well as information regarding the format of the repository 
and instructions for reproducing the results.
- Final_Report.ipynb: The final report containing a high level overview of the project including key takeaways, 
final results, and a recommendations.
- acquire.py: Contains all code utilized for acquiring the Zillow property data.
- prepare.py: Contains all code utilized for preparing the Zillow property data for exploration and modeling.
- explore.py: Contains functions used in the final report for producing visualizations and statistical test results of key takeaways from exploration.
- model.py: Contains functions used in the final report for producing and evaluating machine learning models.
- clustering.py: Contains functions used for building cluster models.
- get_db_url.py: Used for obtaining the URL needed to access the database.
- _acquire.py: Contains an Acquire class with generalized acquisition code.
- _model.py: Contains a Model class used for keeping track of features, target, hyperparameters, and the model used.
- notebook:
    - wrangle.ipynb: Contains the step by step acquisition and preparation process with details and explanations.
    - explore.ipynb: Contains the step by step exploratory analysis process with details and explanations.
    - model.ipynb: Contains the step by step modeling process with details and explanations.
    - univariate_analysis.py: Contains functions used for univariate analysis.
    - bivariate_analysis.py: Contains functions used for bivariate analysis.

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Project Description](#project-description)
3. [Initial Questions](#initial-questions)
4. [Data Dictionary](#data-dictionary)
5. [Instructions for Reproducing the Results](#instructions-for-reproducing-the-results)
6. [Outline of Project Plan](#outline-of-project-plan)
    1. [Data Acquisition](#data-acquisition)
    2. [Data Preparation](#data-preparation)
    3. [Exploratory Analysis](#exploratory-analysis)
    4. [Modeling](#modeling)

## Project Goals

The goal of this project is to identify drivers of logerror in the Zillow property value estimates and produce a model for predicting the logerror using these drivers.

## Project Description

Zillow is interested in identifying what attributes are driving the logerror in their Zestimates. This is important to help produce better prediction models so that Zillow can remain competitive in the housing market. With this in mind we are interested in producing a model for predicting logerror and identifying actionable recommendations that could help reduce the overall average logerror. Additionally, we are interested in single family/single unit properties with transactions in 2017.

## Initial Questions

- Is there a relationship between property age and logerror?
- Does the zip code a property is located in have a relationship with logerror?
- Is there a relationship between the size of a property and logerror?
- Is there a relationship between the number of bathrooms/bedrooms and logerror?
- Is there a relationship between the tax assessed value of a property and logerror?

## Data Dictionary

| Variable              | Meaning      |
| --------------------- | ------------ |
| calculatedfinishedsquarefeet/square_feet | Calculated total finished living area of the home |
| lotsizesquarefeet/lot_size     | Area of the lot in square feet |
| regionidzip           | Zip code in which the property is located |
| yearbuilt             | The year the principal residence was built |
| property_age          | The age in years of the principal residence |
| taxvaluedollarcnt     | The total tax assessed value of the parcel |
| bathroomcnt           | Number of bathrooms in home including fractional bathrooms |
| bedroomcnt            | Number of bedrooms in home |
| logerror              | The difference between the log of the Zestimate and the log of the sale price |


## Instructions for Reproducing the Results

1. Clone this repository into your local machine using the following command:
```bash
git clone git@github.com:alegarcia-dev/zillow-clustering-project.git
```
2. You will need Pandas, Numpy, Matplotlib, Seaborn, and SKLearn installed on your machine.
3. If you don't have login credentials for the MySQL database hosted at data.codeup.com acquire login credentials.
4. Create a file in the main directory titled "env.py" and put your login credentials in the following format:
```python
username = "your_username"
password = "your_password"
hostname = "data.codeup.com"
```
5. Now you can start a Jupyter Notebook session and execute the code blocks in the Final_Report.ipynb notebook.

## Outline of Project Plan
---
### Data Acquisition

In this phase the zillow property data is acquired from the MySQL database. The dataset is quite large so it would be inefficient to retrieve the data from the database each time we need it. For this reason the data needs to be cached in a csv file. Additionally we must look at our requirements and ensure that our acquired data adheres to our requirements which are that the properties acquired have transactions in 2017. Furthermore, we don't want duplicate properties so we must only obtain the most recent transaction date for any properties with multiple transactions in 2017.

- The wrangle.ipynb notebook in the notebooks directory contains a reproducible step by step process for acquiring the data with details 
and explanations.

- The acquire.py file contains all the data acquisition code used in the final report notebook.

- The _acquire.py file contains a data acquisition class used in the acquire.py file.

**Steps Taken:**
1. Create an SQL query to select all properties with transactions in 2017. For properties with more than one transaction in 2017 select only the most recent transaction. All columns associated with the property should be selected.
2. Analyze the columns selected and ensure that there are no duplicate columns. Remove any id columns that won't be useful.
3. Encapsulate all acquisition code in an acquire.py file.

### Data Preparation

In this phase the zillow property data is prepared so that it will be ready for exploration and modeling. The dataset initially contains a large number of missing values. To keep things simple due to time constraints we simply try to drop as many nulls as we can while still maintaining a reasonable percentage of the initial data. In this phase we also need to filter the data to include only single family/single unit properties.

Preparing the data allows us to see through the noise and focus on the data that is useful to the problem at hand.

- The prepare.ipynb notebook in the notebooks directory contains a reproducible step by step process for preparing the data with details and explanations.

- The prepare.py file contains all the data preparation functions used in the final report notebook.

**Steps Taken:**
1. Analyze the data to determine how many null values are present.
2. Remove columns that are missing too many values to be useful.
3. Remove rows with missing values.
4. Remove any properties that are not single family/single unit properties based on the property type description.
5. Encapsulate all preparation code in a prepare.py file.

### Exploratory Analysis

In this phase the zillow property data is analyzed in order to determine which features are most relevant for predicting logerror. Prior to beginning this phase the dataset is split into a train set, validate set, and a test set to maintain the integrity of our process by keeping some data as unseen.

We start by analyzing the distributions of values for each feature. Then we analyze the relation between each feature and the target variable. Finally, we look at how various features interact with each other to determine if any new features can be engineered from existing ones to provide better insights.

- The explore.ipynb notebook in the notebooks directory contains a reproducible step by step process for exploring the data with details and explanations.

- The explore.py file contains all the data exploration functions used in the final report notebook.

- The univariate_analysis.py file in the notebooks directory contains functions used for univariate analysis in exploration.

- The bivariate_analysis.py file in the notebooks directory contains functions used for bivariate analysis in exploration.

- The clustering.py file contains functions used when building clustering models.

**Steps Taken:**
1. Acquire, prepare, and split the data. Use only the train data in exploration.
2. Observe the distributions of values for all variables and identify features with outliers.
3. Using initial questions answer questions using bivariate analysis by observing how the variables interact with the target.
4. Observe the interactions of multiple variables to identify which features may be useful for clustering.
5. Build a clustering model and use it to create a new feature.
6. Observe the results of the clustering model.
7. One hot encode the cluster feature.
8. Perform any additional feature engineering.
9. Update prepare.py file and encapsulate key exploration takeaways in explore.py.

### Modeling

In this phase a regression model is produced for predicting logerror for properties in the zillow property dataset. Several models are created and compared to determine which features and strategies provide the best model performance. The best model will be used in the final report to predict the target variable in the test dataset.

- The model.ipynb notebook in the notebooks directory contains a reproducible step by step process for producing regression models with details and explanations.

- The model.py file contains all the modeling functions used in the final report notebook.

- The _model.py file contains a class used for keeping track of all the features, target, hyperparameters and algorithm used for each machine learning model.

**Steps Taken:**
1. Scale all datasets and remove outliers from train.
2. Establish a baseline model
3. Produce four models without using the cluster feature.
4. Produce four models with the cluster feature.
5. Analyze the results of each model on train and validate.

---

[Back to top](#what-is-driving-the-errors-in-zestimates)