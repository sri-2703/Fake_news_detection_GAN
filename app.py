from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("model/fake_news_model.h5")

# Load SAME TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form["news"]

        vec = vectorizer.transform([news_text]).toarray()
        pred = model.predict(vec)[0][0]

        confidence = round(pred * 100, 2)

        if pred >= 0.5:
            prediction = "REAL NEWS "
        else:
            prediction = "FAKE NEWS "

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text
    )

if __name__ == "__main__":
    app.run(debug=True)
