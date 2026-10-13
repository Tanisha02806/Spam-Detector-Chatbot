import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Training data
messages = [
    "Win a free prize now",
    "Claim your free reward",
    "Congratulations you won cash",
    "Get a special offer today",
    "Please attend the meeting tomorrow",
    "Can we meet after class",
    "Your assignment is due Friday",
    "Please send me the notes",
]
labels = ["Spam", "Spam", "Spam", "Spam", "Not Spam", "Not Spam", "Not Spam", "Not Spam"]

# 2. Text -> numbers (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Words:", sorted(vectorizer.vocabulary_))
print("Matrix shape:", X.shape)

# 3. Train the classifier
model = MultinomialNB()
model.fit(X, labels)

# 4. Quick sanity check
for text in ["Win cash today", "Please send the meeting notes"]:
    pred = model.predict(vectorizer.transform([text]))[0]
    print(f"{text!r} -> {pred}")

# 5. Save the vectorizer and model for testing and Flask
joblib.dump({"vectorizer": vectorizer, "model": model}, "spam_model.joblib")
print("Saved spam_model.joblib")