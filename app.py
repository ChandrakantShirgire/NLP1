import re, json, joblib
import streamlit as st

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️")

def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t).lower())
    t = re.sub(r"http\S+|#url_\w+#|#email_\w+#|#phone_\w+#", " ", t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", t)).strip()

@st.cache_resource
def load():
    return joblib.load("model.joblib")

model = load()
st.title("Fake Job Posting Detector")
st.caption("TF-IDF + Logistic Regression trained on the EMSCAD job postings dataset.")

title = st.text_input("Job title")
desc = st.text_area("Job description", height=160)
req = st.text_area("Requirements", height=100)

if st.button("Check posting"):
    text = clean(f"{title} {desc} {req}")
    if not text:
        st.warning("Enter some text first.")
    else:
        p = float(model.predict_proba([text])[0][1])
        if p >= 0.5:
            st.error(f"Looks FAKE ({p:.0%} fraud probability)")
        else:
            st.success(f"Looks REAL ({1 - p:.0%} confidence)")
        st.progress(p, text="Fraud probability")

with st.expander("Model comparison"):
    m = json.load(open("metrics.json"))
    rows = {k: {x: round(v[x], 3) for x in ("accuracy", "precision", "recall", "f1")} for k, v in m.items() if k != "_meta"}
    st.table(rows)
