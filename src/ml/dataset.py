"""Датасет UCI «Predict Students' Dropout and Academic Success»: скачивание и загрузка."""
import io
import logging
import zipfile
from pathlib import Path

import httpx2
import pandas as pd

logger = logging.getLogger(__name__)

URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "students.csv"
SOURCE = {
    "name": "Predict Students' Dropout and Academic Success",
    "url": "https://doi.org/10.24432/C5MC89",
    "authors": "Realinho V., Vieira Martins M., Machado J., Baptista L. (2021)",
    "license": "CC BY 4.0",
}

# Исходные имена колонок → имена в проекте
COLUMNS = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance": "daytime",
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mother_qualification",
    "Father's qualification": "father_qualification",
    "Mother's occupation": "mother_occupation",
    "Father's occupation": "father_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_up_to_date",
    "Gender": "gender",
    "Scholarship holder": "scholarship",
    "Age at enrollment": "age",
    "International": "international",
    **{
        f"Curricular units {order} sem ({original})": f"sem{n}_{name}"
        for n, order in ((1, "1st"), (2, "2nd"))
        for original, name in (("credited", "credited"), ("enrolled", "enrolled"), ("evaluations", "evaluations"),
                               ("approved", "approved"), ("grade", "grade"), ("without evaluations", "no_evaluations"))
    },
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": "outcome",
}
OUTCOMES = {"Dropout": "dropout", "Enrolled": "enrolled", "Graduate": "graduate"}

_cache = None


def download():
    """Скачивает архив, приводит колонки к именам проекта и сохраняет data/students.csv."""
    logger.info("Скачиваю датасет: %s", URL)
    response = httpx2.get(URL, follow_redirects=True, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        raw = archive.read("data.csv")
    # В файле BOM в начале, разделитель «;», у «Daytime/evening attendance» в имени хвостовая табуляция
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if unknown := set(df.columns) - COLUMNS.keys():
        raise ValueError(f"В датасете новые колонки: {sorted(unknown)}")
    df = df.rename(columns=COLUMNS)
    df["outcome"] = df["outcome"].map(OUTCOMES)
    df.insert(0, "student_id", range(1, len(df) + 1))
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    logger.info("Датасет сохранён: %s строк → %s", len(df), DATA_PATH)
    return df


def load_dataset():
    """Датасет из data/students.csv; при первом обращении скачивается. Результат не изменять — он общий."""
    global _cache
    if _cache is None:
        _cache = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else download()
    return _cache


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download()
