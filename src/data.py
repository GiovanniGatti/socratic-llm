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


class DPOExample(BaseModel):
    prompt: str
    a: DPOEvaluation
    b: DPOEvaluation

    @computed_field
    @property
    def chosen(self) -> Optional[str]:
        if self.a.evaluation_error is not None or self.b.evaluation_error is not None:
            return None
        a_score = self.a.evaluation.summary_score()
        b_score = self.b.evaluation.summary_score()

        chosen: DPOEvaluation
        chosen = self.a if a_score >= b_score else self.b
        return chosen.output

    @computed_field
    @property
    def rejected(self) -> Optional[str]:
        if self.a.evaluation_error is not None or self.b.evaluation_error is not None:
            return None
        a_score = self.a.evaluation.summary_score()
        b_score = self.b.evaluation.summary_score()

        rejected: DPOEvaluation
        rejected = self.b if a_score >= b_score else self.a
        return rejected.output


class TrainDataset(RootModel):
    root: List[DPOExample] = []

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def get_valid(self) -> List[Example]:
        return [e for e in self.root if e.a.evaluation_error is None and e.b.evaluation_error is None]
