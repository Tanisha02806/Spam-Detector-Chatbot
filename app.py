from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model saved by test_model.py
bundle = joblib.load("spam_model.joblib")
vectorizer = bundle["vectorizer"]
model = bundle["model"]

classes = list(model.classes_)            # ['Not Spam', 'Spam']
SPAM = classes.index("Spam")
HAM = classes.index("Not Spam")

analyzer = vectorizer.build_analyzer()    # same tokenizer the model uses

# Per-word evidence: positive = pushes toward Spam, negative = toward Not Spam
word_evidence = model.feature_log_prob_[SPAM] - model.feature_log_prob_[HAM]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("message", "").strip()

    if not text:
        return jsonify(error="Please enter a message."), 400

    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]
    prediction = model.predict(vec)[0]

    # Break the message into words and show how each one influenced the result
    tokens = []
    for word in analyzer(text):
        idx = vectorizer.vocabulary_.get(word)
        tokens.append({
            "word": word,
            "known": idx is not None,
            "evidence": float(word_evidence[idx]) if idx is not None else 0.0,
        })

    return jsonify(
        message=text,
        prediction=str(prediction),
        spam_prob=float(proba[SPAM]),
        ham_prob=float(proba[HAM]),
        tokens=tokens,
    )


if __name__ == "__main__":
    app.run(debug=True)