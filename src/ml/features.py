"""Признаки модели: группы по этапам учёбы, названия и справочники кодов."""
import json
from pathlib import Path

_TABLES = json.loads((Path(__file__).parent / "codes.json").read_text(encoding="utf-8"))

# Признак → справочник «код → название»; у матери и отца справочники общие
CODES = {
    field: {int(code): title for code, title in _TABLES[table].items()}
    for field, table in {
        "marital_status": "marital_status",
        "application_mode": "application_mode",
        "course": "course",
        "previous_qualification": "previous_qualification",
        "nationality": "nationality",
        "mother_qualification": "parent_qualification",
        "father_qualification": "parent_qualification",
        "mother_occupation": "parent_occupation",
        "father_occupation": "parent_occupation",
    }.items()
}

TITLES = {
    "marital_status": "Семейное положение",
    "application_mode": "Способ поступления",
    "application_order": "Приоритет заявления: 0 — первый выбор, 9 — последний",
    "course": "Программа обучения",
    "daytime": "Форма обучения: 1 — дневная, 0 — вечерняя",
    "previous_qualification": "Образование до поступления",
    "previous_grade": "Балл предыдущего образования, 0–200",
    "nationality": "Гражданство",
    "mother_qualification": "Образование матери",
    "father_qualification": "Образование отца",
    "mother_occupation": "Профессия матери",
    "father_occupation": "Профессия отца",
    "admission_grade": "Вступительный балл, 0–200",
    "displaced": "Переехал ради учёбы: 1 — да, 0 — нет",
    "special_needs": "Особые образовательные потребности: 1 — да, 0 — нет",
    "debtor": "Долг перед вузом: 1 — да, 0 — нет",
    "tuition_up_to_date": "Обучение оплачено вовремя: 1 — да, 0 — нет",
    "gender": "Пол: 1 — мужской, 0 — женский",
    "scholarship": "Стипендия: 1 — да, 0 — нет",
    "age": "Возраст при поступлении",
    "international": "Иностранный студент: 1 — да, 0 — нет",
    "unemployment_rate": "Безработица в Португалии в год поступления, %",
    "inflation_rate": "Инфляция в год поступления, %",
    "gdp": "Рост ВВП в год поступления, %",
    "outcome": "Исход: dropout — отчислен, enrolled — ещё учится, graduate — выпустился",
}
for n in (1, 2):
    TITLES |= {
        f"sem{n}_credited": f"{n}-й семестр: предметов перезачтено",
        f"sem{n}_enrolled": f"{n}-й семестр: записан на предметы",
        f"sem{n}_evaluations": f"{n}-й семестр: аттестаций (попыток сдать)",
        f"sem{n}_approved": f"{n}-й семестр: сдано предметов",
        f"sem{n}_grade": f"{n}-й семестр: средний балл по сданным, 0–20",
        f"sem{n}_no_evaluations": f"{n}-й семестр: предметов без аттестации",
    }

CATEGORICAL = ["marital_status", "application_mode", "course", "previous_qualification",
               "mother_qualification", "father_qualification", "mother_occupation", "father_occupation"]

# Известно в день зачисления
ENROLLMENT = ["marital_status", "application_mode", "application_order", "course", "daytime",
              "previous_qualification", "previous_grade", "mother_qualification", "father_qualification",
              "mother_occupation", "father_occupation", "admission_grade", "displaced", "special_needs",
              "debtor", "tuition_up_to_date", "scholarship", "age",
              "unemployment_rate", "inflation_rate", "gdp"]
SEMESTER_FIELDS = ["credited", "enrolled", "evaluations", "approved", "grade", "no_evaluations"]
SEMESTER_1 = [f"sem1_{f}" for f in SEMESTER_FIELDS]
SEMESTER_2 = [f"sem2_{f}" for f in SEMESTER_FIELDS]

# Три момента прогноза — у каждого своя модель
STAGES = {
    "enrollment": ENROLLMENT,
    "semester_1": ENROLLMENT + SEMESTER_1,
    "semester_2": ENROLLMENT + SEMESTER_1 + SEMESTER_2,
}
STAGE_TITLES = {"enrollment": "При поступлении", "semester_1": "После 1-го семестра", "semester_2": "После 2-го семестра"}

OUTCOMES = ["dropout", "enrolled", "graduate"]
OUTCOME_TITLES = {"dropout": "отчислят", "enrolled": "ещё учится", "graduate": "выпустится"}

# Есть в датасете, но в модель не идут: прогноз не должен зависеть от пола и гражданства
EXCLUDED = ["gender", "nationality", "international"]
