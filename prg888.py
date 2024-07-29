import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv('hea.csv')

# Replace '?' with NaN and drop rows with NaN values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

X = df.drop('sex', axis=1)
y = df['sex']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
model = GaussianNB()
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
accu = accuracy_score(y_test, y_pred)
print(f'accuracy: {accu:.2f}')

print("classification_report:")
print(classification_report(y_test, y_pred))

print("confusion_matrix:")
print(confusion_matrix(y_test, y_pred))
