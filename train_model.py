import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import clean_text

# create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# load dataset
data = pd.read_csv("dataset/cb_multi_labeled_balanced.csv")

# keep required columns
data = data[["text", "label"]]

# convert labels to binary
data["label"] = data["label"].apply(
    lambda x: 0 if x == "not_cyberbullying" else 1
)

# clean text
data["clean_text"] = data["text"].apply(clean_text)

X = data["clean_text"]
y = data["label"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save model
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Model saved successfully.")