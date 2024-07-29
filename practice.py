import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

data=pd.read_csv("heart.csv")[["age","sex","cp","chol","heartdisease"]]
print(data.head())

model=BayesianNetwork([('age','heartdisease'),('sex','heartdisease'),('cp','heartdisease'),('chol','heartdisease')])
model.fit(data,estimator=MaximumLikelihoodEstimator)

inference=VariableElimination(model)
evidence={'age':67,'sex':1,'cp':4,'chol':250}
ans=inference.query(variables=['heartdisease'],evidence=evidence)
print(ans)
