import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load dataset
df = pd.read_csv("../data/news.csv")
df = df.dropna()
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

X = df["text"].values
y = df["label"].values

# Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X).toarray()

# 🔑 SAVE VECTORIZER
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced", classes=classes, y=y_train
)
class_weights = dict(zip(classes, class_weights))

# Model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=16,
    class_weight=class_weights,
    validation_data=(X_test, y_test)
)

model.save("fake_news_model.h5")

print("✅ Model and vectorizer saved")
