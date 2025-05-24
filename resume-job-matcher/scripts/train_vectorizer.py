# scripts/train_vectorizer.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

df = pd.read_csv("resume-job-matcher/data/job.csv")
vectorizer = TfidfVectorizer(stop_words='english')
job_matrix = vectorizer.fit_transform(df['description'])


joblib.dump(vectorizer, "resume-job-matcher/models/tfidf_vectorizer.pkl")
joblib.dump(job_matrix, "resume-job-matcher/models/job_matrix.pkl")
