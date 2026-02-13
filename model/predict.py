import pickle
from tensorflow.keras.models import load_model

model = load_model("fake_news_model.h5")

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_news(text):
    text_vec = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_vec)[0][0]

    if prediction >= 0.5:
        return "🟢 REAL NEWS"
    else:
        return "🔴 FAKE NEWS"

while True:
    news = input("\nEnter news text (or type exit): ")
    if news.lower() == "exit":
        break

    print("Prediction:", predict_news(news))
