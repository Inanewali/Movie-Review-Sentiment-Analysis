# sentiment_analysis_imdb.py
# Author: Syed Waliullah

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.datasets import imdb

# 1. Load IMDB dataset (pre-tokenized as integers)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# 2. Decode integers back to words for TF-IDF
word_index = imdb.get_word_index()
index_to_word = {v+3: k for k, v in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"

def decode_review(encoded):
    return " ".join([index_to_word.get(i, "?") for i in encoded])

X_train_text = [decode_review(x) for x in X_train]
X_test_text = [decode_review(x) for x in X_test]

# 3. Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# 4. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Predictions
y_pred = model.predict(X_test_vec)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - IMDB Sentiment Analysis")
plt.show()

# 7. Test custom reviews
custom_review = [
    "The movie was absolutely fantastic, I loved it!",
    "It was a waste of time, very boring and predictable."
]
custom_vec = vectorizer.transform(custom_review)
pred = model.predict(custom_vec)
for txt, label in zip(custom_review, pred):
    print(f"Review: {txt} -> Prediction: {'Positive' if label == 1 else 'Negative'}")
