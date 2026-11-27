"""Restore this repository's artifacts in a fresh process without fitting a model."""

import json
import pickle
from pathlib import Path

import joblib
import numpy as np

from grade_tracker import GradeTracker


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def load_pickle(filename):
    # Only load artifacts produced by save_examples.py or published by this project.
    with (ARTIFACTS / filename).open("rb") as file:
        return pickle.load(file)


def main():
    fruits = load_pickle("fruits.pkl")
    scores = load_pickle("scores.pkl")
    combined = load_pickle("complex_data.pkl")
    assert combined == {"fruits": fruits, "scores": scores, "active": True}
    print(f"Restored list and dictionaries: {combined}")

    grades = load_pickle("course_grades.pkl")
    assert isinstance(grades, GradeTracker) and grades.grades == scores
    print(f"Restored {grades.course_name}: mean {grades.average():.1f}; top student {grades.top_student()}")
    grades.add_grade("Student D", 96)
    with (ARTIFACTS / "course_grades_updated.pkl").open("wb") as file:
        pickle.dump(grades, file, protocol=pickle.HIGHEST_PROTOCOL)
    updated = load_pickle("course_grades_updated.pkl")
    assert updated == grades and updated.top_student() == ("Student D", 96)
    print(f"Updated tracker: mean {updated.average():.1f}; top student {updated.top_student()}")

    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    hours = np.asarray(metrics["example_hours"]).reshape(-1, 1)
    pickle_predictions = load_pickle("study_hours_model.pkl").predict(hours)
    joblib_predictions = joblib.load(ARTIFACTS / "study_hours_model.joblib").predict(hours)
    np.testing.assert_allclose(pickle_predictions, metrics["predictions_before_saving"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(joblib_predictions, pickle_predictions, rtol=0, atol=1e-12)
    print("Hours | Pickle | Joblib")
    for hour, pickle_score, joblib_score in zip(hours.ravel(), pickle_predictions, joblib_predictions):
        print(f"{hour:5.0f} | {pickle_score:6.2f} | {joblib_score:6.2f}")
    print("Both restored models reproduce the predictions saved before serialization.")


if __name__ == "__main__":
    main()
