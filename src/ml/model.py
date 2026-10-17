"""Обучение трёх моделей — по одной на момент прогноза, — сохранение и история обучений."""
import datetime
import json
import logging
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.explain import typical_values
from src.ml.features import CATEGORICAL, OUTCOMES, STAGES

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
HISTORY_PATH = MODELS_DIR / "training_history.json"

# Тип модели → (класс, параметры по умолчанию); переданные в /model/train гиперпараметры их дополняют
MODEL_TYPES = {
    "logreg": (LogisticRegression, {"max_iter": 3000, "class_weight": "balanced"}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 300, "class_weight": "balanced",
                                               "n_jobs": -1, "random_state": 42}),
    "boosting": (HistGradientBoostingClassifier, {"class_weight": "balanced", "random_state": 42}),
}


def build_pipeline(columns, model_type, hyperparameters):
    classifier, defaults = MODEL_TYPES[model_type]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), [c for c in columns if c not in CATEGORICAL]),
        # коды, которые встречаются реже 10 раз, сливаются в одну категорию «редкие»
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False),
         [c for c in columns if c in CATEGORICAL]),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier(**{**defaults, **hyperparameters}))])


def evaluate(pipeline, x, y):
    """Метрики на тестовой выборке."""
    proba = pipeline.predict_proba(x)
    classes = list(pipeline.classes_)
    predicted = [classes[i] for i in proba.argmax(axis=1)]
    recall = recall_score(y, predicted, labels=OUTCOMES, average=None)
    return {
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "f1_macro": round(float(f1_score(y, predicted, average="macro")), 4),
        "auc_dropout": round(float(roc_auc_score(y == "dropout", proba[:, classes.index("dropout")])), 4),
        "recall": {outcome: round(float(r), 4) for outcome, r in zip(OUTCOMES, recall)},
        # строки — настоящий исход, столбцы — прогноз; порядок: dropout, enrolled, graduate
        "confusion": confusion_matrix(y, predicted, labels=OUTCOMES).tolist(),
        "test_size": len(y),
    }


def train(df, model_type="logreg", hyperparameters=None):
    """Обучает по модели на каждый момент прогноза. Выборка делится 80/20 одинаково для всех трёх."""
    hyperparameters = hyperparameters or {}
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["outcome"], random_state=42)
    stages, metrics = {}, {}
    for stage, columns in STAGES.items():
        pipeline = build_pipeline(columns, model_type, hyperparameters)
        stages[stage] = pipeline.fit(train_df[columns], train_df["outcome"])
        metrics[stage] = evaluate(pipeline, test_df[columns], test_df["outcome"])
    logger.info("Обучена модель %s: AUC отчисления %s", model_type,
                " → ".join(str(m["auc_dropout"]) for m in metrics.values()))
    return {
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "stages": stages,
        "metrics": metrics,
        "typical": typical_values(train_df),
        "test_ids": [int(i) for i in test_df["student_id"]],
    }


def save_model(bundle):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)


def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # файл от другой версии scikit-learn — обучим заново
        logger.warning("Не удалось загрузить %s: %s", MODEL_PATH, exc)
        return None


def load_history():
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return []


def save_history(bundle):
    history = load_history()
    history.append({
        "timestamp": bundle["trained_at"],
        "model_type": bundle["model_type"],
        "hyperparameters": bundle["hyperparameters"],
        "metrics": {stage: {k: m[k] for k in ("accuracy", "f1_macro", "auc_dropout")}
                    for stage, m in bundle["metrics"].items()},
    })
    MODELS_DIR.mkdir(exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
