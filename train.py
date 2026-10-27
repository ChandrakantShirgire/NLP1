import re, json, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t).lower())
    t = re.sub(r"http\S+|#url_\w+#|#email_\w+#|#phone_\w+#", " ", t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", t)).strip()

df = pd.read_csv("fake_job_postings.csv")
df["text"] = (df.title.fillna("") + " " + df.description.fillna("") + " " + df.requirements.fillna("")).map(clean)
df = df[df.text.str.len() > 0]
df = df.groupby("fraudulent", group_keys=False).sample(frac=2000/len(df), random_state=42)
Xtr, Xte, ytr, yte = train_test_split(df.text, df.fraudulent, test_size=0.2, stratify=df.fraudulent, random_state=42)

models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=10),
}
res, pipes = {}, {}
for name, clf in models.items():
    p = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3)), ("clf", clf)])
    p.fit(Xtr, ytr)
    pred = p.predict(Xte)
    res[name] = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred), recall=recall_score(yte, pred),
                     f1=f1_score(yte, pred), cm=confusion_matrix(yte, pred).tolist())
    pipes[name] = p
    print(name, {k: (round(v, 4) if k != "cm" else v) for k, v in res[name].items()})

res["_meta"] = dict(rows=len(df), fake=int(df.fraudulent.sum()), train=len(Xtr), test=len(Xte), test_fake=int(yte.sum()),
                    baseline_acc=float(1 - yte.mean()))
json.dump(res, open("metrics.json", "w"), indent=2)
joblib.dump(pipes["Logistic Regression"], "model.joblib", compress=3)
print(res["_meta"])
