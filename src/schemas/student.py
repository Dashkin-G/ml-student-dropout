"""Схемы запросов и ответов API."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from src.ml.features import CATEGORICAL, CODES, SEMESTER_FIELDS, TITLES

Flag = Literal[0, 1]
Stage = Literal["enrollment", "semester_1", "semester_2"]
Outcome = Literal["dropout", "enrolled", "graduate"]
RiskLevel = Literal["high", "medium", "low"]

# Студент № 2192 из тестовой выборки: при поступлении риск 8 %, после двух несданных семестров — 99 %
EXAMPLE = {
    "marital_status": 1, "application_mode": 1, "application_order": 6, "course": 9500, "daytime": 1,
    "previous_qualification": 1, "previous_grade": 138.0, "mother_qualification": 4, "father_qualification": 19,
    "mother_occupation": 2, "father_occupation": 3, "admission_grade": 124.7, "displaced": 1, "special_needs": 0,
    "debtor": 0, "tuition_up_to_date": 1, "scholarship": 0, "age": 18,
    "unemployment_rate": 16.2, "inflation_rate": 0.3, "gdp": -0.92,
    "sem1_credited": 0, "sem1_enrolled": 7, "sem1_evaluations": 0, "sem1_approved": 0, "sem1_grade": 0.0,
    "sem1_no_evaluations": 0,
    "sem2_credited": 0, "sem2_enrolled": 8, "sem2_evaluations": 0, "sem2_approved": 0, "sem2_grade": 0.0,
    "sem2_no_evaluations": 0,
}


def feature(name, **limits):
    return Field(description=TITLES[name], **limits)


def semester(name, upper):
    return Field(None, ge=0, le=upper, description=TITLES[name])


class StudentFeatures(BaseModel):
    """Признаки студента. Поля семестра передаются все шесть или ни одного: без семестров прогноз —
    на момент поступления, с одним 1-м семестром — после него."""
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE})

    marital_status: int = feature("marital_status")
    application_mode: int = feature("application_mode")
    application_order: int = feature("application_order", ge=0, le=9)
    course: int = feature("course")
    daytime: Flag = feature("daytime")
    previous_qualification: int = feature("previous_qualification")
    previous_grade: float = feature("previous_grade", ge=0, le=200)
    mother_qualification: int = feature("mother_qualification")
    father_qualification: int = feature("father_qualification")
    mother_occupation: int = feature("mother_occupation")
    father_occupation: int = feature("father_occupation")
    admission_grade: float = feature("admission_grade", ge=0, le=200)
    displaced: Flag = feature("displaced")
    special_needs: Flag = feature("special_needs")
    debtor: Flag = feature("debtor")
    tuition_up_to_date: Flag = feature("tuition_up_to_date")
    scholarship: Flag = feature("scholarship")
    age: int = feature("age", ge=15, le=80)
    unemployment_rate: float = feature("unemployment_rate")
    inflation_rate: float = feature("inflation_rate")
    gdp: float = feature("gdp")
    sem1_credited: Optional[int] = semester("sem1_credited", 40)
    sem1_enrolled: Optional[int] = semester("sem1_enrolled", 40)
    sem1_evaluations: Optional[int] = semester("sem1_evaluations", 60)
    sem1_approved: Optional[int] = semester("sem1_approved", 40)
    sem1_grade: Optional[float] = semester("sem1_grade", 20)
    sem1_no_evaluations: Optional[int] = semester("sem1_no_evaluations", 40)
    sem2_credited: Optional[int] = semester("sem2_credited", 40)
    sem2_enrolled: Optional[int] = semester("sem2_enrolled", 40)
    sem2_evaluations: Optional[int] = semester("sem2_evaluations", 60)
    sem2_approved: Optional[int] = semester("sem2_approved", 40)
    sem2_grade: Optional[float] = semester("sem2_grade", 20)
    sem2_no_evaluations: Optional[int] = semester("sem2_no_evaluations", 40)

    @field_validator(*CATEGORICAL)
    @classmethod
    def known_code(cls, value, info: ValidationInfo):
        if value not in CODES[info.field_name]:
            raise ValueError(f"неизвестный код {value}, справочник — GET /dataset/codes")
        return value

    @model_validator(mode="after")
    def complete_semesters(self):
        filled = []
        for n in (1, 2):
            values = [getattr(self, f"sem{n}_{f}") for f in SEMESTER_FIELDS]
            if None in values and any(v is not None for v in values):
                raise ValueError(f"{n}-й семестр: передайте все шесть полей sem{n}_* или ни одного")
            filled.append(None not in values)
            if filled[-1] and getattr(self, f"sem{n}_approved") > getattr(self, f"sem{n}_enrolled"):
                raise ValueError(f"{n}-й семестр: сдано больше предметов, чем записано")
        if filled[1] and not filled[0]:
            raise ValueError("2-й семестр передан без 1-го")
        return self


class Probabilities(BaseModel):
    dropout: float
    enrolled: float
    graduate: float


class StagePrediction(BaseModel):
    stage: Stage
    title: str
    probabilities: Probabilities
    outcome: Outcome = Field(description="Самый вероятный исход")


class Reason(BaseModel):
    feature: str
    text: str = Field(description="Как у студента, например «сдано 1 из 6 во 2-м семестре»")
    typical: str = Field(description="Как у типичного студента, например «обычно 5 из 6»")
    raises_risk: bool
    risk_if_typical: float = Field(description="Риск отчисления, если бы этот признак был как у типичного студента")
    shift: float = Field(description="Сдвиг шансов отчисления (логит): чем больше по модулю, тем сильнее причина")


class PredictionResponse(BaseModel):
    stages: List[StagePrediction] = Field(description="Прогноз на каждый момент, для которого хватает признаков")
    risk: float = Field(description="Риск отчисления на последний известный момент")
    reasons: List[Reason] = Field(description="До трёх признаков, сильнее всего сдвигающих риск")


class BatchRequest(BaseModel):
    students: List[StudentFeatures] = Field(description="Студенты группы: признаки — как у POST /predict")


class BatchStudent(BaseModel):
    index: int = Field(description="Номер студента в запросе, с нуля")
    stage: Stage = Field(description="Последний момент, для которого хватает признаков")
    risk: float = Field(description="Риск отчисления на этот момент")
    level: RiskLevel = Field(description="Уровень риска: high — от 50 %, medium — 25–50 %, low — до 25 %")
    outcome: Outcome = Field(description="Самый вероятный исход")


class BatchResponse(BaseModel):
    count: int
    levels: Dict[RiskLevel, int] = Field(description="Сколько студентов на каждом уровне риска")
    high_share: float = Field(description="Доля студентов с высоким риском")
    students: List[BatchStudent] = Field(description="Студенты по убыванию риска")


class TrainingConfig(BaseModel):
    model_type: Literal["logreg", "random_forest", "boosting"] = "logreg"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict,
                                            description="Параметры конструктора модели scikit-learn")


class StudentRecord(BaseModel):
    student_id: int
    split: Literal["train", "test"] = Field(description="test — модель этого студента не видела")
    outcome: Outcome = Field(description="Настоящий исход из датасета")
    features: StudentFeatures


class Example(BaseModel):
    student_id: int
    title: str
    course: int
    outcome: Outcome
    risk_start: float = Field(description="Риск при поступлении")
    risk_end: float = Field(description="Риск после 2-го семестра")


class ErrorResponse(BaseModel):
    code: int
    message: str
    details: str
