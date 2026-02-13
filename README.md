📰 Fake News Detection System using GAN

📌 Abstract
Fake news has become a major challenge in the digital era due to the rapid spread of misinformation through social media and online platforms.  
This project presents a **Fake News Detection System** that uses **Natural Language Processing (NLP)** and **Deep Learning** techniques to classify news articles as **Real** or **Fake**.  

A **Generative Adversarial Network (GAN)** concept is incorporated to enhance the robustness of the classifier by improving data representation.  
The system also includes a **Flask-based web interface** for real-time prediction.

---

 🎯 Objectives
- Detect fake news accurately using machine learning
- Use GAN concepts to strengthen model learning
- Provide a user-friendly web interface
- Visualize model performance using graphs and confusion matrix

---

 🧠 Technologies Used
- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Flask**
- **HTML & CSS**
- **Git & GitHub**

---

 🗂️ Project Folder Structure
FakeNews/
├── app.py
├── model/
│ ├── fake_news_model.h5
│ └── tfidf_vectorizer.pkl
├── data/
│ └── news.csv
├── templates/
│ └── index.html
├── static/
│ └── style.css
├── README.md

---

 🧩 System Architecture Diagram

User Input (News Text)
|
v
Web Interface (HTML + CSS)
|
v
Flask Backend
|
v
TF-IDF Vectorizer
|
v
Neural Network Classifier
|
v
Prediction Result
(✔ Real News / ✖ Fake News)

---

 🤖 GAN Architecture (Conceptual Diagram)

  Random Noise (z)
         |
         v
  +----------------+
  |   Generator    |
  | (Fake Samples) |
  +----------------+
         |
 Generated Fake Data
         |
         v
+-------------------------+
| Discriminator |
| Real or Fake Decision |
+-------------------------+
^ ^
| |
Real News Generated News

---

 🧠 GAN Explanation (Theory)

A **Generative Adversarial Network (GAN)** consists of two neural networks:

 🔹 Generator
- Generates synthetic (fake) data
- Attempts to fool the discriminator

 🔹 Discriminator
- Distinguishes between real and fake data
- Acts as a binary classifier

 🔹 Objective Function
\[
\min_G \max_D V(D,G) =
\mathbb{E}_{x \sim p_{data}}[\log D(x)] +
\mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
\]

In this project, GAN concepts help in **improving data diversity and model robustness**, especially when datasets are limited.

---

🔎 Methodology

1. Data collection and cleaning  
2. Text preprocessing using TF-IDF  
3. Train neural network classifier  
4. Apply class balancing techniques  
5. Evaluate using accuracy and confusion matrix  
6. Deploy model using Flask web application  

---

 📊 Model Evaluation

🔹 Accuracy Graph
The accuracy graph shows improvement in training and validation accuracy as epochs increase, indicating effective learning.

🔹 Confusion Matrix

| Actual / Predicted | Fake | Real |
|-------------------|------|------|
| Fake              | TN   | FP   |
| Real              | FN   | TP   |

- **TP**: Real news correctly predicted  
- **TN**: Fake news correctly predicted  
- **FP**: Fake predicted as real  
- **FN**: Real predicted as fake  

---

🌐 Web Application Features
- Text input for news content
- Real-time prediction
- ✔ Tick mark for Real News
- ✖ Cross mark for Fake News
- Confidence score display
- Clean and modern UI

---

✅ Sample Inputs
Real News
Government announces new education policy for digital learning

Fake News
Aliens secretly controlling Indian government says viral post

---

📈 Results
Achieved good accuracy on balanced dataset
Correctly distinguishes real and fake news
Robust performance due to TF-IDF and class balancing

---

🚀 Future Enhancements
Use transformer models like BERT
Multilingual fake news detection
Social media-based fake news analysis
Deployment on cloud platforms
Real-time news scraping integration

---

🎓 Conclusion
The Fake News Detection System effectively identifies misinformation using NLP and deep learning techniques.
By integrating GAN concepts and a web interface, the system becomes more robust, scalable, and user-friendly.
