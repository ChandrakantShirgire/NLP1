---
title: "Fake Job Posting Detector"
subtitle: "NLP, Machine Learning and Cloud Computing defaulter project"
---

**ISBM College of Engineering, Nande, Pune | AI and Data Science | TE Div B | 2026-27 Sem-I**

## 1. Aim

Build a one-page app that says whether a job advertisement looks real or fake, and compare Naive Bayes with Logistic Regression.

## 2. Data and NLP

- Dataset: Real or Fake Job Posting Prediction (EMSCAD, Kaggle). 2,000 postings sampled (stratified) from 17,880, 97 fake (4.9%).
- Job title, description and requirements were joined into one text column.
- Cleaning: lowercase, HTML tags, URLs and placeholder tokens removed, only letters kept, English stop words dropped.
- TF-IDF with unigrams and bigrams, 20,000 features, words seen in fewer than 3 postings ignored.
- Split 80:20, stratified. Train 1,600 rows, test 400 rows (19 fake).

## 3. Models and comparison

Naive Bayes (alpha 0.1) and Logistic Regression (C = 10, balanced class weights) were trained on the same TF-IDF features. Scores below are on the fake class of the test set.

| Model | Accuracy | Precision | Recall | F1 |
|:--|:-:|:-:|:-:|:-:|
| Naive Bayes | 0.965 | 0.778 | 0.368 | 0.500 |
| Logistic Regression | 0.980 | 0.867 | 0.684 | 0.765 |

| Confusion matrix | TN | FP | FN | TP |
|:--|:-:|:-:|:-:|:-:|
| Naive Bayes | 379 | 2 | 12 | 7 |
| Logistic Regression | 379 | 2 | 6 | 13 |

**Better model: Logistic Regression.** It has the higher F1 (0.765 against 0.500) and catches 13 of 19 fake postings, against 7 for Naive Bayes. Both wrongly flag only 2 real jobs, but Naive Bayes misses almost two thirds of the scams. For a scam detector, missing a fake posting costs more than a false alarm, so recall matters more.

**Why accuracy alone is not enough.** Only 4.8% of postings are fake. A model that calls every posting real scores 95.3% accuracy and catches no scam. Both models show 96 to 98% accuracy, which hides the real difference: recall on fake postings is 37% against 68%. Precision, recall and F1 on the fake class give the true picture.

## 4. Cloud deployment

The app is a single Streamlit file (`app.py`) with the trained model saved as `model.joblib`. It is pushed to GitHub and published on Streamlit Community Cloud.

- **IaaS** gives raw servers or virtual machines; you install and manage everything (AWS EC2).
- **PaaS** gives a ready platform; you only supply code (Streamlit Community Cloud, Heroku).
- **SaaS** gives a finished application to use (Gmail, Google Docs).

This project uses **PaaS**: we upload code and the platform runs it, with no server set-up.

Live link: ____________________________ (or see screenshots attached)

## 5. Conclusion

TF-IDF with Logistic Regression detects fake job postings better than Naive Bayes on this imbalanced data (F1 0.765, recall 0.684). The test set has only 19 fake postings, so scores can shift with a different split. The app should be used as a warning, not a final verdict.
