from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

data = [
    ("I love this movie", "positive"),
    ("This film was amazing", "positive"),
    ("I actually enjoyed it", "positive"),
    ("I hated that movie", "negative"),
    ("This film was terrible", "negative"),
    ("I did not like it", "negative")
]


text,label = zip(*data)

vect = CountVectorizer()
X = vect.fit_transform(text)

clf = MultinomialNB()
clf.fit(X,label)


test = [
    "I love this film",
    "I hated the movie",
    "It was an awesome movie",
    "This movie was not good"
]


X_test = vect.transform(test)


y_pred = clf.predict(X_test)

print(test)
print(y_pred)