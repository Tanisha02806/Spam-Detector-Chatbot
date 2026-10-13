# 📩 Spam Detector Chatbot Using Naive Bayes

An interactive web-based spam detection chatbot that uses the **Multinomial Naive Bayes** algorithm to classify messages as **Spam** or **Not Spam**. The application also visualizes *why* the model made its decision by showing the spam probability and the influence of each word.

## 📌 Project Overview

Spam detection is a common text-classification problem in machine learning. A system must decide whether an incoming message is unwanted (spam) or legitimate (not spam) based on the words it contains.

This project trains a Naive Bayes classifier on labeled messages, evaluates it on unseen test data, and serves it through a Flask web application. Users can type a message into a chat interface and see the prediction, the probability of each class, and a word-by-word breakdown of the evidence.

## ✨ Features

- **Chat Interface:** Check messages in a chatbot-style window.
- **Spam / Not Spam Verdict:** Instant classification of any message.
- **Probability Bar:** See the model's confidence for each class.
- **Word Influence Visualizer:** Words are coloured by how strongly they push toward Spam or Not Spam.
- **Unknown Word Detection:** Words missing from the training data are shown in grey and ignored by the model.
- **Model Evaluation:** Train/test split with accuracy, confusion matrix and classification report.
- **Saved Model:** The trained model is stored with joblib and reused by the web app.
- **REST API Integration:** A Flask `/predict` endpoint connects the frontend to the model.
- **Input Validation:** Empty messages return a readable error.

## 🛠️ Technologies Used

| Technology   | Purpose                                |
|--------------|----------------------------------------|
| Python       | Model training and backend logic       |
| scikit-learn | CountVectorizer and MultinomialNB      |
| Flask        | Backend REST API and page serving      |
| joblib       | Saving and loading the trained model   |
| HTML / CSS   | Chat interface layout and styling      |
| JavaScript   | Sending requests and rendering results |

## 🧠 Naive Bayes Algorithm

Naive Bayes is a probabilistic classifier based on **Bayes' theorem**. It estimates the probability that a message belongs to a class given the words it contains:

```
P(class | words) ∝ P(class) × P(word1 | class) × P(word2 | class) × ... × P(wordn | class)
```

Where:

- **P(class):** How common the class is in the training data (prior).
- **P(word | class):** How often the word appears in messages of that class (likelihood).

The model is called "naive" because it assumes every word contributes independently of the others. The class with the higher score is the prediction.

### Text Representation

Messages are converted to numbers using a **bag-of-words** model (`CountVectorizer`). Each message becomes a vector of word counts, and word order is ignored.

### Word Influence (Visualizer)

For each word, the app calculates:

```
evidence = log P(word | Spam) - log P(word | Not Spam)
```

- **Positive value:** the word pushes toward Spam (shown in red).
- **Negative value:** the word pushes toward Not Spam (shown in green).
- **Larger magnitude:** stronger influence (more intense colour).
- **Unknown word:** not in the training vocabulary, so it is ignored (shown in grey).

### How the System Works

1. Collect labeled messages (Spam / Not Spam).
2. Split the data into training and testing sets.
3. Fit the vectorizer on the training messages to build the vocabulary.
4. Train the Naive Bayes model on the training vectors.
5. Evaluate the model on the unseen test set.
6. Retrain on all the data and save the model to disk.
7. Load the saved model in Flask when the server starts.
8. For each new message, vectorize it and return the prediction, probabilities and per-word evidence.

## 📊 Algorithm Complexity

Let:

- **N** be the number of training messages.
- **L** be the average number of words per message.
- **V** be the vocabulary size.

| Phase      | Time Complexity | Notes                                                          |
|------------|-----------------|----------------------------------------------------------------|
| Training   | O(N × L)        | Each word in each message is counted once                      |
| Prediction | O(L)            | Words are looked up in the vocabulary and scored for 2 classes |
| Space      | O(V)            | Stores word probabilities per class                            |

Naive Bayes is fast to train and predict, which makes it a good baseline for text classification.

## 📈 Model Results

On the held-out test set (12 messages), the model achieved about **92% accuracy**, with 11 of 12 messages classified correctly. Results depend on the small hand-written dataset and may change if the data or random split changes.

## 🏗️ Project Structure

```
spam-detector/
├── app.py              # Flask backend and /predict API
├── test_model.py       # Trains, evaluates and saves the model
├── spam_model.joblib   # Saved model (created by test_model.py)
├── templates/
│   └── index.html      # Chat UI and word visualizer
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible version supported by your dependencies
- pip
- Git (optional, for version control)

### 1. Clone the Repository

```bash
git clone https://github.com/Tanisha02806/spam-detector.git
cd spam-detector
```

Replace the URL with your actual repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it.

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install flask scikit-learn joblib
```

### 4. Train and Test the Model

```bash
python test_model.py
```

This prints the accuracy, confusion matrix and classification report, then creates `spam_model.joblib`.

### 5. Start the Web App

```bash
python app.py
```

The app should be available at:

- Web interface: http://127.0.0.1:5000

Open the URL in your browser and start typing messages.

## 🔌 API Documentation

### Predict Message

**Endpoint:** `POST /predict`

**Purpose:** Receives a message and returns the Naive Bayes prediction, class probabilities and per-word evidence.

**Example request:**

```json
{
  "message": "Win cash today"
}
```

**Example response:**

```json
{
  "message": "Win cash today",
  "prediction": "Spam",
  "spam_prob": 0.9886,
  "ham_prob": 0.0114,
  "tokens": [
    { "word": "win", "known": true, "evidence": 1.80 },
    { "word": "cash", "known": true, "evidence": 1.96 },
    { "word": "today", "known": true, "evidence": 0.70 }
  ]
}
```

In the response:

- `prediction` is either `Spam` or `Not Spam`.
- `spam_prob` and `ham_prob` add up to 1.
- `known` is `false` for words not in the training vocabulary.
- `evidence` is positive for spam-leaning words and negative for not-spam-leaning words.

**Error response (empty message), status 400:**

```json
{ "error": "Please enter a message." }
```

**Test with PowerShell:**

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"message":"Win cash today"}'
```

## 🧪 Testing

The following scenarios should be tested to verify the correctness and reliability of the application:

| Test Case                                        | Expected Result                                           |
|--------------------------------------------------|-----------------------------------------------------------|
| Message with clear spam words ("Win cash today") | Predicted as Spam with high spam probability              |
| Normal message ("Please send me the notes")      | Predicted as Not Spam                                     |
| Message with words unseen in training            | Unknown words are shown in grey and ignored               |
| Empty message                                    | A readable error is shown and nothing is sent             |
| Message containing HTML (`<b>hi</b>`)            | Displayed as plain text, not rendered as HTML             |
| Flask server stopped                             | The frontend shows a "could not reach the server" message |
| Repeated messages                                | Each message gets its own result in the chat              |
| Missing `spam_model.joblib`                      | The app fails to start, so run `test_model.py` first      |

Record actual test outcomes before submitting the project. The table describes expected behavior, not a claim that every test has been independently verified.

## ⚠️ Limitations

- The dataset is small (48 messages), so the model knows only a limited vocabulary.
- Words not seen during training are ignored, which can lead to mistakes on ordinary messages. For example, "Happy birthday hope you have a great day" was wrongly classified as Spam during testing.
- Bag-of-words ignores word order, context and sarcasm.

## 🚀 Future Improvements

Possible extensions, if required, include:

- Training on a larger real-world dataset such as the SMS Spam Collection.
- Using TF-IDF features and comparing other models (Logistic Regression, SVM).
- Removing stop words and adding stemming.
- A feedback button to correct wrong predictions and retrain the model.
- Deployment of the application to a hosting service.

## 🎯 Learning Outcomes

Through this project, the following concepts can be practiced:

- Text classification using Naive Bayes.
- Bag-of-words feature extraction.
- Train/test splitting and avoiding data leakage.
- Model evaluation with accuracy, confusion matrix, precision and recall.
- Saving and loading machine learning models.
- REST API development using Flask.
- Frontend and backend integration.
- Explaining model decisions through visualization.

## 👩‍💻 Author

**Tanisha Thakur**

GitHub: [Tanisha02806](https://github.com/Tanisha02806)

Built to demonstrate spam detection with Naive Bayes through an interactive, explainable chatbot application.