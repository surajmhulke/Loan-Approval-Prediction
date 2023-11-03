# Loan-Approval-Prediction
Loan Approval Prediction using Machine Learning
Table of Contents

    Introduction
    Importing Libraries and Dataset
    Data Preprocessing and Visualization
    Splitting Dataset
    Model Training and Evaluation
    Conclusion

Introduction

Loans are a major requirement of the modern world. Banks profit significantly from providing loans. It helps students manage education and living expenses, and enables people to purchase luxuries like houses and cars.

However, deciding whether an applicant's profile is relevant for a loan is a complex task. Banks need to consider various factors. In this project, we use Machine Learning with Python to predict the relevance of a candidate's profile for loan approval. We consider key features such as Marital Status, Education, Applicant Income, and Credit History.

You can download the dataset used in this project here.
Importing Libraries and Dataset

We start by importing necessary libraries for data manipulation and visualization. We load the dataset using pandas.

python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("LoanApprovalPrediction.csv")

Data Preprocessing and Visualization

    We handle categorical variables and visualize unique values in columns.
    We use Label Encoder to convert categorical values to numerical.
    We visualize the correlation between different features.
    We use Catplot to visualize the Gender and Marital Status of applicants.

Splitting Dataset

We split the dataset into training and testing sets using scikit-learn's train_test_split.

python

from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

Model Training and Evaluation

We train various classifiers (KNeighborsClassifier, RandomForestClassifier, SVC, LogisticRegression) and evaluate their accuracy scores on both the training and testing datasets.

python

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    print("Accuracy score of ", clf.__class__.__name__, "=", 100 * metrics.accuracy_score(Y_train, Y_pred))
    
for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print("Accuracy score of ", clf.__class.__name__, "=", 100 * metrics.accuracy_score(Y_test, Y_pred))

Conclusion

Random Forest Classifier provides the best accuracy (82%) for the testing dataset. For even better results, ensemble learning techniques like Bagging and Boosting can be explored.
