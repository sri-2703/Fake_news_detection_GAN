import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data
df = pd.read_csv("../data/news.csv")
df = df.dropna()
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

X = df["text"].values
y = df["label"].values

# Load model & vectorizer
model = load_model("fake_news_model.h5")
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X_vec = vectorizer.transform(X).toarray()

# Predictions
y_pred = (model.predict(X_vec) >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

disp.plot()
plt.title("Confusion Matrix - Fake News Detection")
plt.show()
