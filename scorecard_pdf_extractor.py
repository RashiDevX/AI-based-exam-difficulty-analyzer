import pdfplumber
import pandas as pd
import re

def extract_scorecard_rows(pdf_file, student_id, max_marks=100):
    rows = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                match = re.search(
                    r"([A-Za-z][A-Za-z\s]+)\s*[:\-]?\s*(\d{1,3})",
                    line
                )

                if match:
                    subject = match.group(1).strip()
                    marks = int(match.group(2))

                    if 0 <= marks <= max_marks:
                        rows.append({
                            "student_id": student_id,
                            "subject": subject,
                            "marks": marks
                        })

    return pd.DataFrame(rows)
