#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yunxin
"""

from __future__ import print_function
from builtins import range
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import plotly.express as px

st.title("Iris Dataset")
st.subheader('Overview')
st.markdown("Yunxin Liu - HW1")
st.markdown("This is a classification model performed on the Iris dataset - one of the best known dataset in the world of data science. The major question I'm trying to answer with the classification model is as follows: Also which of the petal/sepal measurements are more useful features to look at?")

st.sidebar.header('User Input Parameters')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.7)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names.reshape(1,3))

models = []
pred = []

models.append(('Decision Tree', DecisionTreeClassifier(max_depth = 3, random_state = 1)))
models.append(('Gaussian Naive Bayes', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 3)))
models.append(('Logistic Regression', LogisticRegression(solver = 'newton-cg')))
models.append(('Linear SVC', SVC(kernel='linear')))

for name, model in models:
    pred.append(iris.target_names[model.fit(X,Y).predict(df)])

idx = [models[i][0] for i in range(len(models))]
pred = pd.DataFrame(pred, columns = ['Prediction'], index = idx)

st.subheader('Prediction for each classifier')
st.write(pred)

st.subheader('Prediction from the majority of classifiers')
st.write(pred['Prediction'].value_counts().index[0])

st.subheader('Reflection')
st.markdown("This one I used a simple classification. I hope next class I'll be able to use other more advanced modeling.")

#code credits to https://github.com/terryz1/Iris_Classification/blob/master/Iris_demo_app.py