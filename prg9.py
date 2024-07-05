import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt


heartdisease = pd.read_csv('heart.csv')
heartdisease = heartdisease.replace('?', np.nan)

print('Few examples from the dataset are given below')
print(heartdisease.head())


discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
continuous_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


heartdisease[continuous_vars] = discretizer.fit_transform(heartdisease[continuous_vars])


model = BayesianNetwork([('age', 'trestbps'), ('age', 'fbs'),
                         ('sex', 'trestbps'), ('exang', 'trestbps'),
                         ('trestbps', 'heartdisease'), ('fbs', 'heartdisease'),
                         ('heartdisease', 'restecg'), ('heartdisease', 'thalach'),
                         ('chol', 'heartdisease')])

print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartdisease, estimator=MaximumLikelihoodEstimator)

print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)


evidence_age = discretizer.transform(np.array([[30, 0, 0, 0, 0]]))[0][0]

evidence_chol = discretizer.transform(np.array([[0, 0, 254, 0, 0]]))[0][2]

print('\n1. Probability of HeartDisease given Age=30')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': int(evidence_age)})
print(q)

print('\n2. Probability of HeartDisease given cholesterol=254')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': int(evidence_chol)})
print(q)