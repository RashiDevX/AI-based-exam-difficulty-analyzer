import streamlit as st
import pandas as pd
import pickle

from preprocess import preprocess_text
from scorecard_pdf_extractor import extract_scorecard_rows

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Exam Difficulty Analyzer",
    layout="centered"
)

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    with open("model/difficulty_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- APP TITLE ----------------
st.title("📘 Exam Difficulty Analyzer")

# ---------------- MODE SELECTION ----------------
mode = st.radio(
    "Select Analysis Mode",
    ("Question-Based Difficulty", "Scorecard-Based Exam Difficulty")
)

# ==================================================
# 🧠 QUESTION-BASED DIFFICULTY
# ==================================================
if mode == "Question-Based Difficulty":

    st.subheader("🧠 Question-Based Difficulty (ML)")

    question = st.text_area(
        "Enter an exam question",
        placeholder="e.g. Analyze deadlock prevention techniques"
    )

    if st.button("Analyze Difficulty"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            processed = preprocess_text(question)
            prediction = model.predict([processed])[0]
            st.success(f"Predicted Difficulty: **{prediction}**")

# ==================================================
# 📊 SCORECARD-BASED EXAM DIFFICULTY
# ==================================================
else:
    st.subheader("📊 Scorecard-Based Exam Difficulty")

    uploaded_files = st.file_uploader(
        "Upload scorecard PDFs (same exam)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        all_rows = []

        for idx, pdf in enumerate(uploaded_files):
            student_id = f"S{idx + 1}"
            df_pdf = extract_scorecard_rows(pdf, student_id)

            if df_pdf is not None and not df_pdf.empty:
                all_rows.append(df_pdf)

        if not all_rows:
            st.error("No marks could be extracted from the uploaded PDFs.")
        else:
            combined_df = pd.concat(all_rows, ignore_index=True)

            st.subheader("📄 Extracted Subject Marks")
            st.dataframe(combined_df)

            st.subheader("📊 Subject-wise Difficulty")
            subject_difficulties = []

            for subject, group in combined_df.groupby("subject"):
                scores = group["marks"]
                avg = scores.mean()
                pass_rate = (scores >= 40).mean() * 100
                std_dev = scores.std() if len(scores) > 1 else 0

                if avg >= 70 and pass_rate >= 80:
                    difficulty = "Easy"
                elif avg >= 40 and pass_rate >= 60 and std_dev <= 25:
                    difficulty = "Moderate"
                else:
                    difficulty = "Difficult"

                subject_difficulties.append(difficulty)

                st.write(
                    f"**{subject}** → {difficulty} | "
                    f"Avg: {avg:.2f} | Pass: {pass_rate:.1f}% | Std: {std_dev:.2f}"
                )

            st.subheader("📝 Overall Exam Difficulty")

            if subject_difficulties.count("Difficult") >= subject_difficulties.count("Easy"):
                exam_difficulty = "Difficult"
            elif subject_difficulties.count("Easy") > subject_difficulties.count("Moderate"):
                exam_difficulty = "Easy"
            else:
                exam_difficulty = "Moderate"

            st.success(f"Final Exam Difficulty: **{exam_difficulty}**")
