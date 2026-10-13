import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- 1. Bigger dataset ----------
spam = [
    "Win a free prize now",
    "Claim your free reward",
    "Congratulations you won cash",
    "Get a special offer today",
    "Free entry to win a brand new phone",
    "You have been selected for a cash prize",
    "Urgent claim your reward now",
    "Limited time offer buy one get one free",
    "Congratulations you won a free vacation",
    "Click here to claim your free gift",
    "Earn money fast working from home",
    "Exclusive deal just for you act now",
    "You won a lottery claim your cash now",
    "Get free recharge today only",
    "Special discount offer ends tonight",
    "Win big cash prizes every day",
    "Your account has won a free reward",
    "Hurry limited offer free gift waiting",
    "Claim your free cash bonus today",
    "Lowest price guaranteed buy now and win",
    "Congratulations you are our lucky winner",
    "Free trial offer click now to get cash",
    "Get paid to win free prizes",
    "Act now to receive your special reward",
]
ham = [
    "Please attend the meeting tomorrow",
    "Can we meet after class",
    "Your assignment is due Friday",
    "Please send me the notes",
    "Are you coming to lunch today",
    "Let me know when you reach home",
    "The project report is ready for review",
    "Can you share the lecture slides",
    "Dinner at my place on Saturday",
    "I will call you after the meeting",
    "Don't forget to submit the form",
    "What time does the class start tomorrow",
    "Thanks for helping me with the homework",
    "The exam schedule has been updated",
    "Can we reschedule our meeting to Monday",
    "Please review the document and reply",
    "I am running late see you soon",
    "Did you finish the assignment",
    "Let's study together this weekend",
    "Happy birthday hope you have a great day",
    "The teacher moved the test to next week",
    "Could you send me the meeting notes",
    "I will bring the books tomorrow",
    "See you in class on Monday",
]

messages = spam + ham
labels = ["Spam"] * len(spam) + ["Not Spam"] * len(ham)

# ---------- 2. Split into train and test ----------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    messages, labels,
    test_size=0.25,        # 25% held back for testing
    random_state=42,       # same split every run
    stratify=labels,       # keep spam/not-spam ratio equal in both sets
)
print("Training messages:", len(X_train_text))
print("Testing messages: ", len(X_test_text))

# ---------- 3. Vectorize (fit on TRAIN only) ----------
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_text)   # learns vocabulary
X_test = vectorizer.transform(X_test_text)         # only transforms

# ---------- 4. Train ----------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------- 5. Evaluate on the unseen test set ----------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion matrix (rows = actual, columns = predicted):")
print("Order:", list(model.classes_))
print(confusion_matrix(y_test, y_pred, labels=model.classes_))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# ---------- 6. See each test message and the model's answer ----------
print("Test results:")
for text, actual, pred in zip(X_test_text, y_test, y_pred):
    mark = "OK " if actual == pred else "WRONG"
    print(f"[{mark}] actual={actual:<8} predicted={pred:<8} | {text}")

# ---------- 7. Try your own messages ----------
print("\nCustom messages:")
for text in ["Win cash today", "Meeting is at 5pm", "Free gift click now", "Send me the notes"]:
    pred = model.predict(vectorizer.transform([text]))[0]
    print(f"{text!r} -> {pred}")

# ---------- 8. Retrain on ALL data and save for Flask ----------
final_vectorizer = CountVectorizer()
X_all = final_vectorizer.fit_transform(messages)
final_model = MultinomialNB()
final_model.fit(X_all, labels)
joblib.dump({"vectorizer": final_vectorizer, "model": final_model}, "spam_model.joblib")
print("\nSaved spam_model.joblib (trained on all data)")
