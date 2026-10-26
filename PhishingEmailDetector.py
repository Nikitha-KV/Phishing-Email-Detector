"""
Phishing Email Detector
Simple cybersecurity project using TF-IDF + Logistic Regression.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "phishing_email.csv"

# Small fallback dataset so the project can be tested immediately.
demo_data = pd.DataFrame({
    "text": [
        "Your bank account is locked. Click this link immediately to verify your password.",
        "Congratulations! You won a prize. Send your OTP to claim the reward.",
        "Urgent: your account will be suspended unless you confirm your login details.",
        "You have received a secure message. Verify your account using the attached link.",
        "Please submit your assignment before Friday. Contact me if you have questions.",
        "Your college timetable has been updated. Please check the student portal.",
        "Meeting is scheduled for tomorrow at 10 AM in the seminar hall.",
        "Your library book is due next week. Please return it on time.",
        "The project review will be conducted on Monday. Bring your presentation.",
        "Thank you for registering for the workshop. Your participation is confirmed."
    ],
    "label": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
})

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    # Try common column names.
    text_col = next((c for c in ["text", "body", "email", "message"] if c in df.columns), None)
    label_col = next((c for c in ["label", "Label", "class", "target"] if c in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError("Dataset must contain a text column and a label column.")

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    # Convert common text labels to 0/1.
    if df["label"].dtype == "object":
        df["label"] = df["label"].astype(str).str.lower().map(
            lambda x: 1 if x in ["1", "phishing", "spam", "malicious", "yes"] else 0
        )
else:
    print("phishing_email.csv not found. Using the built-in demo dataset.")
    df = demo_data

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nModel Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\n--- Phishing Email Detector ---")
email = input("Paste an email/message to check: ")

prediction = model.predict([email])[0]
probability = model.predict_proba([email])[0][1]

if prediction == 1:
    print("\nRESULT: POSSIBLE PHISHING EMAIL")
    print("Risk score:", round(probability * 100, 2), "%")
    print("Advice: Do not click links or share passwords, OTPs, or personal information.")
else:
    print("\nRESULT: LIKELY LEGITIMATE EMAIL")
    print("Risk score:", round(probability * 100, 2), "%")
    print("Advice: Still verify the sender before opening links or attachments.")
