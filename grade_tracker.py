from dataclasses import dataclass, field
from statistics import mean


@dataclass
class GradeTracker:
    course_name: str
    grades: dict[str, float] = field(default_factory=dict)

    def add_grade(self, student: str, grade: float) -> None:
        if not 0 <= grade <= 100:
            raise ValueError("Grades must be between 0 and 100")
        self.grades[student] = grade

    def average(self) -> float | None:
        return mean(self.grades.values()) if self.grades else None

    def top_student(self) -> tuple[str, float] | None:
        return max(self.grades.items(), key=lambda item: item[1], default=None)
