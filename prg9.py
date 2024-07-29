import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

data=pd.read_csv('heart.csv')[['sex','age','heartdisease']]

model=BayesianNetwork([('sex','heartdisease'),('age','heartdisease')])
model.fit(data,estimator=MaximumLikelihoodEstimator)

inference=VariableElimination(model)
evidence={'sex':1,'age':67}

ans=inference.query(variables=['heartdisease'],evidence=evidence)

print(ans)
