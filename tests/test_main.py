import pytest
from fastapi.testclient import TestClient

from src.ml import model
from src.schemas.student import EXAMPLE

ENROLLMENT_ONLY = {k: v for k, v in EXAMPLE.items() if not k.startswith("sem")}
WITHOUT_SEM2 = {k: v for k, v in EXAMPLE.items() if not k.startswith("sem2")}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Модель и история обучений — во временной папке, чтобы тесты не трогали models/
    models_dir = tmp_path_factory.mktemp("models")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(model, "MODELS_DIR", models_dir)
        mp.setattr(model, "MODEL_PATH", models_dir / "model.joblib")
        mp.setattr(model, "HISTORY_PATH", models_dir / "training_history.json")
        from src.main import app
        with TestClient(app) as test_client:  # with — чтобы при старте обучилась модель
            yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "dataset_available": True}


def test_dataset_info(client):
    info = client.get("/dataset/info").json()
    assert info["rows"] == 4424
    assert sum(info["outcomes"].values()) == 4424


def test_codes(client):
    assert len(client.get("/dataset/codes").json()["course"]["codes"]) == 17


def test_predict_three_stages(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    result = response.json()
    assert [s["stage"] for s in result["stages"]] == ["enrollment", "semester_1", "semester_2"]
    for s in result["stages"]:
        assert sum(s["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    # студент № 2192: при поступлении риск низкий, после двух несданных семестров — высокий
    assert result["stages"][0]["probabilities"]["dropout"] < 0.5 < result["risk"]
    assert result["reasons"][0]["feature"] == "sem2_approved"
    assert result["reasons"][0]["raises_risk"]


def test_predict_by_known_semesters(client):
    assert [s["stage"] for s in client.post("/predict", json=ENROLLMENT_ONLY).json()["stages"]] == ["enrollment"]
    assert len(client.post("/predict", json=WITHOUT_SEM2).json()["stages"]) == 2


def test_more_passed_lowers_risk(client):
    before = client.post("/predict", json=EXAMPLE).json()["risk"]
    after = client.post("/predict", json={**EXAMPLE, "sem2_approved": 7, "sem2_grade": 13.0}).json()["risk"]
    assert after < before


@pytest.mark.parametrize("body, field", [
    ({**EXAMPLE, "course": 1234}, "course"),
    ({**ENROLLMENT_ONLY, "sem1_enrolled": 6}, "1-й семестр"),
    ({**ENROLLMENT_ONLY, **{k: v for k, v in EXAMPLE.items() if k.startswith("sem2")}}, "без 1-го"),
    ({**EXAMPLE, "sem2_approved": 9}, "больше предметов"),
    ({}, "course"),
])
def test_predict_validation(client, body, field):
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == 422
    assert field in response.json()["details"]


def test_predict_batch(client):
    response = client.post("/predict/batch", json={"students": [ENROLLMENT_ONLY, EXAMPLE, WITHOUT_SEM2]})
    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 3 and sum(result["levels"].values()) == 3
    risks = [s["risk"] for s in result["students"]]
    assert risks == sorted(risks, reverse=True)
    # каждый студент — на своём последнем моменте и с тем же риском, что у POST /predict
    assert {s["index"]: s["stage"] for s in result["students"]} == {0: "enrollment", 1: "semester_2", 2: "semester_1"}
    single = client.post("/predict", json=EXAMPLE).json()["risk"]
    assert next(s["risk"] for s in result["students"] if s["index"] == 1) == single


@pytest.mark.parametrize("students", [[], [EXAMPLE] * 501])
def test_predict_batch_size(client, students):
    response = client.post("/predict/batch", json={"students": students})
    assert response.status_code == 422
    assert "students" in response.json()["details"]


def test_students(client):
    examples = client.get("/students/examples").json()
    assert len(examples) == 6
    record = client.get(f"/students/{examples[0]['student_id']}").json()
    assert record["split"] == "test"
    assert client.post("/predict", json=record["features"]).status_code == 200
    assert client.get("/students/99999").status_code == 404


def test_train(client):
    response = client.post("/model/train", json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 50}})
    assert response.status_code == 200
    assert set(response.json()["metrics"]) == {"enrollment", "semester_1", "semester_2"}
    assert client.get("/model/status").json()["model_type"] == "random_forest"
    assert len(client.get("/model/metrics").json()) == 2  # обучение при старте + это


def test_train_errors(client):
    assert client.post("/model/train", json={"model_type": "xgboost"}).status_code == 422
    response = client.post("/model/train", json={"model_type": "logreg", "hyperparameters": {"depth": 3}})
    assert response.status_code == 400
    assert "depth" in response.json()["message"]
