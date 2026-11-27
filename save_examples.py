"""Create the lesson's pickle and joblib files and a regression plot."""

import json
import pickle
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from grade_tracker import GradeTracker

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
EXAMPLE_HOURS = np.array([[5.0], [10.0], [15.0], [20.0]])


def save_pickle(filename, value):
    with (ARTIFACTS / filename).open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    fruits = ["apple", "pear", "mango"]
    scores = {"Student A": 92, "Student B": 84, "Student C": 88}
    save_pickle("fruits.pkl", fruits)
    save_pickle("scores.pkl", scores)
    save_pickle("complex_data.pkl", {"fruits": fruits, "scores": scores, "active": True})

    grades = GradeTracker("Data Engineering")
    for student, score in scores.items():
        grades.add_grade(student, score)
    save_pickle("course_grades.pkl", grades)

    rng = np.random.default_rng(26723820)
    hours = rng.uniform(1, 25, 120).reshape(-1, 1)
    test_scores = 48 + 1.9 * hours.ravel() + rng.normal(0, 4, len(hours))
    train_hours, test_hours, train_y, test_y = train_test_split(
        hours, test_scores, test_size=0.25, random_state=42
    )
    model = LinearRegression().fit(train_hours, train_y)
    test_predictions = model.predict(test_hours)
    save_pickle("study_hours_model.pkl", model)
    joblib.dump(model, ARTIFACTS / "study_hours_model.joblib", compress=3)

    metrics = {
        "scikit_learn_version": sklearn.__version__,
        "train_samples": len(train_y),
        "test_samples": len(test_y),
        "test_mse": float(mean_squared_error(test_y, test_predictions)),
        "test_r2": float(r2_score(test_y, test_predictions)),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "example_hours": EXAMPLE_HOURS.ravel().tolist(),
        "predictions_before_saving": model.predict(EXAMPLE_HOURS).tolist(),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(metrics, indent=2))

    grid = np.linspace(1, 25, 100).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.7), layout="constrained")
    ax.scatter(train_hours.ravel(), train_y, color="#42718a", alpha=0.45, label="Training samples")
    ax.scatter(test_hours.ravel(), test_y, color="#bf6d26", alpha=0.75, label="Held-out samples")
    ax.plot(grid.ravel(), model.predict(grid), color="#182e43", linewidth=2, label="OLS prediction")
    ax.set(xlabel="Study hours per week", ylabel="Synthetic scores", title="Study hours and simulated exam scores")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(ARTIFACTS / "study_hours_model.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
