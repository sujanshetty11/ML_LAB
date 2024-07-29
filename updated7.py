from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = [
    ("I love this movie", "positive"),
    ("This film was amazing", "positive"),
    ("I actually enjoyed it", "positive"),
    ("I hated that movie", "negative"),
    ("This film was terrible", "negative"),
    ("I did not like it", "negative")
] 

test= [
    "I love this film",
    "I hated the movie",
    "It was an awesome movie",
    "This movie was not good"
]


x_train,y_train=zip(*data)

vectorizer=CountVectorizer()
x_train=vectorizer.fit_transform(x_train)
x_test=vectorizer.transform(test)

model=MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

for text,label in zip(test,y_pred):
    print(f"text:'{text}' ==> predicted:'{label}'")