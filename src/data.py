from typing import List

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

    def get_score(self) -> float:
        score = 0.
        score += 0.25 if self.questions.lower().strip() == "yes" else 0.
        score += 0.25 if self.reveal_answer.lower().strip() == "no" else 0.
        score += 0.25 / 5 * self.on_topic
        score += 0.25 / 5 * self.helpful
        return score


class Example(BaseModel):
    prompt: str
    output: str
    evaluation: Evaluation

    @computed_field
    @property
    def score(self) -> float:
        return self.evaluation.get_score()


class Scores(RootModel):
    root: List[Example] = []

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]
