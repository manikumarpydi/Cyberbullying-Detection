import pickle
from preprocessing import clean_text

# load model
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_text(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "⚠ Cyberbullying Detected"
    else:
        return "✅ Safe Comment"