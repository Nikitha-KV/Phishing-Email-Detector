# Phishing Email Detector

## Project Overview
A simple cybersecurity project that uses Natural Language Processing (NLP) and Machine Learning to identify whether an email/message is likely to be phishing.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## Dataset
Recommended public phishing-email dataset:
https://www.kaggle.com/competitions/phishing-email-detection

The program can also run with its small built-in demo dataset for testing.

If using your own CSV, save it as `phishing_email.csv` with:
- `text` = email/message content
- `label` = 1 for phishing/spam, 0 for legitimate

## Setup
1. Install Python 3.
2. Open the project folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Optional: place `phishing_email.csv` in the same folder.
5. Run:
   `python PhishingEmailDetector.py`

## How It Works
1. Email text is collected.
2. TF-IDF converts the text into numerical features.
3. Logistic Regression classifies the message.
4. The program displays the result and a risk score.

## Output
The system reports:
- POSSIBLE PHISHING EMAIL, or
- LIKELY LEGITIMATE EMAIL

## Ethical Note
This project is for educational cybersecurity analysis only. It does not guarantee that an email is safe or malicious.
