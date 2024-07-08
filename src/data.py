import statistics
from typing import List, Optional

from pydantic import RootModel, BaseModel, computed_field


class Dataset(RootModel):
    root: List[str] = []

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


class Evaluation(BaseModel):
    questions: str
    on_topic: float
    helpful: float
    reveal_answer: str

    def summary_score(self) -> float:
        score = 0.
        score += 0.25 if self.questions.lower().strip() == "yes" else 0.
        score += 0.25 if self.reveal_answer.lower().strip() == "no" else 0.
        score += 0.25 / 5 * self.on_topic
        score += 0.25 / 5 * self.helpful
        return score


class Example(BaseModel):
    prompt: str
    output: str
    raw_evaluation: str
    evaluation_error: Optional[str]
    evaluation: Optional[Evaluation]

    @computed_field
    @property
    def score(self) -> Optional[float]:
        return self.evaluation.summary_score() if self.evaluation is not None else None


class Scores(RootModel):
    root: List[Example] = []

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def get_valid(self) -> List[Example]:
        return [e for e in self.root if e.evaluation_error is None]

    def avg_summary_score(self) -> float:
        mean = statistics.mean(e.evaluation.summary_score() for e in self.get_valid())
        return round(mean, 2)

    def avg_questions(self) -> float:
        mean = statistics.mean(e.evaluation.questions.lower().strip() == "yes" for e in self.get_valid())
        return round(mean, 2)

    def avg_on_topic(self) -> float:
        mean = statistics.mean(e.evaluation.on_topic for e in self.get_valid())
        return round(mean / 5, 2)

    def avg_helpfulness(self) -> float:
        mean = statistics.mean(e.evaluation.helpful for e in self.get_valid())
        return round(mean / 5, 2)

    def avg_reveal_answer(self) -> float:
        mean = statistics.mean(e.evaluation.reveal_answer.lower().strip() == "yes" for e in self.get_valid())
        return round(mean, 2)


class DPOEvaluation(BaseModel):
    output: str
    raw_evaluation: str
    evaluation_error: Optional[str]
    evaluation: Optional[Evaluation]

    def summary_score(self) -> float:
        return self.evaluation.summary_score() if self.evaluation_error is None else -1.


class DPOExample(BaseModel):
    prompt: str
    chosen_eval: DPOEvaluation
    rejected_eval: DPOEvaluation
    all_evaluations: List[DPOEvaluation] = []

    @computed_field
    @property
    def chosen(self) -> Optional[str]:
        if self.chosen_eval.evaluation_error is not None:
            return None

        return self.chosen_eval.output

    @computed_field
    @property
    def rejected(self) -> Optional[str]:
        if self.rejected_eval.evaluation_error is not None:
            return None

        return self.rejected_eval.output


class TrainDataset(RootModel):
    root: List[DPOExample] = []

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def get_valid(self) -> List[Example]:
        return [
            e for e in self.root
            if e.chosen_eval.evaluation_error is None and e.rejected_eval.evaluation_error is None
        ]

    def get_eligible_for_training(self) -> List[Example]:
        return list(
            filter(
                lambda e: (e.chosen_eval.summary_score() > 0.8 and
                           e.chosen_eval.summary_score() > e.rejected_eval.summary_score()),
                self.get_valid()
            )
        )
