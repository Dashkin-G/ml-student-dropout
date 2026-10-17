"""Прогноз по этапам учёбы и его объяснение: что сильнее всего сдвигает риск отчисления."""
import math

import pandas as pd

from src.ml.features import OUTCOMES, STAGES, STAGE_TITLES


def typical_values(df):
    """Типичный студент обучающей выборки — с ним сравниваем, объясняя прогноз."""
    typical = {"age": float(df["age"].median()), "admission_grade": float(df["admission_grade"].median())}
    for n in (1, 2):
        enrolled, approved = df[f"sem{n}_enrolled"], df[f"sem{n}_approved"]
        typical[f"sem{n}_pass_rate"] = float((approved / enrolled)[enrolled > 0].median())
        typical[f"sem{n}_grade"] = float(df[f"sem{n}_grade"][approved > 0].median())
    return typical


def known_stages(student):
    """Этапы, для которых у студента заполнены все признаки."""
    return [stage for stage, columns in STAGES.items() if all(student.get(c) is not None for c in columns)]


def probabilities(pipeline, rows):
    """Вероятности трёх исходов для каждой строки."""
    proba = pipeline.predict_proba(pd.DataFrame(rows)[list(pipeline.feature_names_in_)])
    classes = list(pipeline.classes_)
    return [{outcome: round(float(p[classes.index(outcome)]), 4) for outcome in OUTCOMES} for p in proba]


def predict(bundle, student):
    """Прогноз на каждый известный этап и причины — для последнего из них."""
    stages = []
    for stage in known_stages(student):
        p = probabilities(bundle["stages"][stage], [student])[0]
        stages.append({"stage": stage, "title": STAGE_TITLES[stage], "probabilities": p, "outcome": max(p, key=p.get)})
    last = stages[-1]
    return {"stages": stages, "risk": last["probabilities"]["dropout"], "reasons": reasons(bundle, student, last["stage"])}


def _number(x):
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}".replace(".", ",")


def _alternatives(student, stage, typical):
    """Варианты «как у типичного студента»: (признак, как есть, как обычно, изменённый студент)."""
    found = []

    def add(feature, now, usual, **changes):
        found.append((feature, now, usual, {**student, **changes}))

    if student["tuition_up_to_date"] == 0:
        add("tuition_up_to_date", "обучение не оплачено вовремя", "обычно оплачено", tuition_up_to_date=1)
    if student["debtor"] == 1:
        add("debtor", "есть долг перед вузом", "обычно долга нет", debtor=0)
    if student["scholarship"] == 1:
        add("scholarship", "получает стипендию", "обычно без стипендии", scholarship=0)
    if student["daytime"] == 0:
        add("daytime", "вечерняя форма обучения", "обычно дневная", daytime=1)
    if abs(student["age"] - typical["age"]) >= 3:
        add("age", f"возраст при поступлении — {student['age']}", f"обычно {_number(typical['age'])}",
            age=typical["age"])
    if abs(student["admission_grade"] - typical["admission_grade"]) >= 10:
        add("admission_grade", f"вступительный балл {_number(student['admission_grade'])} из 200",
            f"обычно {_number(typical['admission_grade'])}", admission_grade=typical["admission_grade"])

    for n in {"enrollment": [], "semester_1": [1], "semester_2": [1, 2]}[stage]:
        enrolled, approved, grade = (student[f"sem{n}_{k}"] for k in ("enrolled", "approved", "grade"))
        if enrolled == 0:
            continue
        where = "в 1-м семестре" if n == 1 else "во 2-м семестре"
        usual_grade = typical[f"sem{n}_grade"]
        usual = round(enrolled * typical[f"sem{n}_pass_rate"])
        if approved != usual:
            # средний балл есть только у тех, кто что-то сдал
            new_grade = 0 if usual == 0 else (grade if approved > 0 else usual_grade)
            add(f"sem{n}_approved", f"сдано {approved} из {enrolled} {where}", f"обычно {usual} из {enrolled}",
                **{f"sem{n}_approved": usual, f"sem{n}_grade": new_grade})
        if approved > 0 and abs(grade - usual_grade) >= 1:
            add(f"sem{n}_grade", f"средний балл {where} — {_number(grade)} из 20", f"обычно {_number(usual_grade)}",
                **{f"sem{n}_grade": usual_grade})
    return found


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def reasons(bundle, student, stage, top=3, min_shift=0.2):
    """Признаки, сильнее всего сдвигающие риск отчисления.

    Каждый признак по очереди ставим «как у типичного студента» и смотрим, каким стал бы риск
    (risk_if_typical). Ранжируем по сдвигу шансов (логит), а не процентных пунктов: у риска 99,99 %
    любая одна замена меняет его меньше чем на пункт, хотя причина настоящая.
    """
    alternatives = _alternatives(student, stage, bundle["typical"])
    if not alternatives:
        return []
    rows = [student] + [changed for *_, changed in alternatives]
    risks = [p["dropout"] for p in probabilities(bundle["stages"][stage], rows)]
    found = []
    for (feature, now, usual, _), risk in zip(alternatives, risks[1:]):
        shift = _logit(risks[0]) - _logit(risk)
        if abs(shift) >= min_shift:
            found.append({"feature": feature, "text": now, "typical": usual, "raises_risk": shift > 0,
                          "risk_if_typical": risk, "shift": round(shift, 3)})
    return sorted(found, key=lambda r: abs(r["shift"]), reverse=True)[:top]


def examples(bundle, df):
    """Показательные студенты из тестовой выборки — модель их не видела."""
    test = df[df["student_id"].isin(bundle["test_ids"])]
    rows = test.to_dict(orient="records")
    start = probabilities(bundle["stages"]["enrollment"], rows)
    end = probabilities(bundle["stages"]["semester_2"], rows)
    table = test[["student_id", "course", "outcome", "sem2_approved"]].assign(
        start=[p["dropout"] for p in start], end=[p["dropout"] for p in end], enrolled_p=[p["enrolled"] for p in end])
    table = table.assign(rise=table["end"] - table["start"])
    dropout, graduate = table[table["outcome"] == "dropout"], table[table["outcome"] == "graduate"]
    picks = [
        # что-то во 2-м семестре сдано и риск не упёрся в 100 % — «что если» есть куда двигать
        ("Высокий риск", dropout[(dropout["sem2_approved"] > 0) & (dropout["end"] < 0.95)], "end", False),
        ("Риск вырос за год", dropout, "rise", False),
        ("Риск снизился за год", graduate, "rise", True),
        ("Ещё учится", table[table["outcome"] == "enrolled"], "enrolled_p", False),
        ("Уверенный выпускник", graduate, "end", True),
        ("Модель ошиблась", graduate, "end", False),
    ]
    chosen, result = set(), []
    for title, candidates, column, ascending in picks:
        row = candidates[~candidates["student_id"].isin(chosen)].sort_values(column, ascending=ascending).iloc[0]
        chosen.add(int(row["student_id"]))
        result.append({"student_id": int(row["student_id"]), "title": title, "course": int(row["course"]),
                       "outcome": row["outcome"], "risk_start": float(row["start"]), "risk_end": float(row["end"])})
    return result
