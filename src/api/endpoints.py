"""HTTP-эндпоинты: датасет, студенты, прогноз и обучение модели."""
import logging
import random
from typing import List

from fastapi import APIRouter, HTTPException, Query

from src.ml import explain, model
from src.ml.dataset import DATA_PATH, SOURCE, load_dataset
from src.ml.features import CODES, EXCLUDED, OUTCOME_TITLES, OUTCOMES, STAGE_TITLES, STAGES, TITLES
from src.schemas.student import Example, PredictionResponse, StudentFeatures, StudentRecord, TrainingConfig

router = APIRouter()
logger = logging.getLogger(__name__)

# Обученная модель и показательные студенты — общие для всех запросов
state = {"model": None, "examples": None}


def startup():
    """Модель с диска; при первом запуске — скачать датасет и обучить логистическую регрессию."""
    bundle = model.load_model()
    if bundle is None:
        bundle = model.train(load_dataset())
        model.save_model(bundle)
        model.save_history(bundle)
    state["model"], state["examples"] = bundle, None


def current_model():
    if state["model"] is None:
        raise HTTPException(status_code=400, detail="Модель не обучена. Обучите её: POST /model/train")
    return state["model"]


@router.get("/health")
async def health():
    return {"status": "ok", "model_loaded": state["model"] is not None, "dataset_available": DATA_PATH.exists()}


@router.get("/dataset/info")
async def dataset_info():
    df = load_dataset()
    counts = df["outcome"].value_counts()
    return {
        "rows": len(df),
        "columns": len(df.columns) - 1,  # без student_id
        "outcomes": {outcome: int(counts[outcome]) for outcome in OUTCOMES},
        "source": SOURCE,
    }


@router.get("/dataset/preview")
async def dataset_preview(n: int = Query(5, ge=1, le=100)):
    return load_dataset().head(n).to_dict(orient="records")


@router.get("/dataset/codes")
async def dataset_codes():
    return {field: {"title": TITLES[field], "codes": codes} for field, codes in CODES.items()}


@router.get("/students/examples", response_model=List[Example])
async def students_examples():
    bundle = current_model()
    if state["examples"] is None:
        state["examples"] = explain.examples(bundle, load_dataset())
    return state["examples"]


@router.get("/students/random", response_model=StudentRecord)
async def random_student():
    """Случайный студент из тестовой выборки — модель его не видела."""
    return await student(random.choice(current_model()["test_ids"]))


@router.get("/students/{student_id}", response_model=StudentRecord)
async def student(student_id: int):
    df = load_dataset()
    rows = df[df["student_id"] == student_id].to_dict(orient="records")
    if not rows:
        raise HTTPException(status_code=404, detail=f"Студента № {student_id} нет: номера от 1 до {len(df)}")
    row = rows[0]
    bundle = state["model"]
    return {
        "student_id": student_id,
        "split": "test" if bundle and student_id in bundle["test_ids"] else "train",
        "outcome": row["outcome"],
        "features": {name: row[name] for name in StudentFeatures.model_fields},
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentFeatures):
    result = explain.predict(current_model(), student.model_dump())
    logger.info("Прогноз: %s, риск %.2f", result["stages"][-1]["stage"], result["risk"])
    return result


@router.post("/model/train")
def train(config: TrainingConfig):
    try:
        bundle = model.train(load_dataset(), config.model_type, config.hyperparameters)
    except (TypeError, ValueError) as exc:  # неизвестный или недопустимый гиперпараметр
        raise HTTPException(status_code=400, detail=f"Не удалось обучить {config.model_type}: {exc}")
    model.save_model(bundle)
    model.save_history(bundle)
    state["model"], state["examples"] = bundle, None
    return {"model_type": bundle["model_type"], "trained_at": bundle["trained_at"], "metrics": bundle["metrics"]}


@router.get("/model/status")
async def model_status():
    bundle = state["model"]
    if bundle is None:
        return {"trained": False}
    return {
        "trained": True,
        "model_type": bundle["model_type"],
        "hyperparameters": bundle["hyperparameters"],
        "trained_at": bundle["trained_at"],
        "metrics": bundle["metrics"],
        "typical": bundle["typical"],  # с ним сравниваются причины прогноза
    }


@router.get("/model/schema")
async def model_schema():
    return {
        "stages": {stage: {"title": STAGE_TITLES[stage], "features": columns} for stage, columns in STAGES.items()},
        "features": {name: TITLES[name] for name in STAGES["semester_2"]},
        "excluded": {name: TITLES[name] for name in EXCLUDED},
        "outcomes": OUTCOME_TITLES,
    }


@router.get("/model/metrics")
async def model_metrics(n: int = Query(5, ge=1, le=100)):
    return model.load_history()[-n:]
