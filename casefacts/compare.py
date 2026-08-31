"""Asking several models the same question, to find out which one to trust.

There is no benchmark for "reads this claim file correctly". The models
disagree in ways that only show up on your own documents: one invents a date,
one refuses a question it could have answered, one reads a smudged page right.
So rather than pick a model on someone else's evaluation, run two or three on
the same question and look at what they do differently.

What is worth comparing, in order:

* **How many findings survived checking.** A model with six findings and two
  unverified is worse than one with four and none.
* **Whether they cite the same pages.** Two models landing on the same page
  from different wordings is the strongest signal available here that the page
  is the right one.
* **What one found that the others missed.** Usually the interesting column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from .answer import Answer, ask
from .index import Index

log = logging.getLogger(__name__)


@dataclass
class Comparison:
    question: str
    answers: list[Answer] = field(default_factory=list)

    def pages_cited(self, answer: Answer) -> set[tuple[str, int]]:
        return {
            (f.doc_id, f.page_no)
            for f in answer.findings
            if f.verdict != "unverified" and f.doc_id
        }

    def agreement(self) -> dict[str, Any]:
        """Where the models landed on the same page, and where they did not."""
        if len(self.answers) < 2:
            return {}
        sets = [self.pages_cited(a) for a in self.answers]
        shared = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        return {
            "pages_all_models_cited": sorted(f"{doc.rsplit('/', 1)[-1]} p.{page}" for doc, page in shared),
            "agreement": round(len(shared) / len(union), 2) if union else 0.0,
            "only_one_model": {
                answer.model: sorted(
                    f"p.{page}"
                    for doc, page in (sets[i] - set.union(*(sets[:i] + sets[i + 1:])))
                )
                for i, answer in enumerate(self.answers)
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answers": [a.to_dict() for a in self.answers],
            "agreement": self.agreement(),
            "summary": [
                {
                    "model": a.model,
                    "seconds": round(a.seconds, 1),
                    "claimed": len(a.findings),
                    "verified": a.verified_count,
                    "unverified": len(a.findings) - a.verified_count,
                    "answer": a.answer,
                }
                for a in self.answers
            ],
        }


def compare(
    index: Index,
    question: str,
    models: Sequence[str],
    **kwargs: Any,
) -> Comparison:
    """Run one question through several models, one after another.

    Sequentially, not in parallel: this machine holds one model in memory at a
    time, and running two at once would swap rather than overlap. Ollama
    unloads the previous model as it loads the next, so the first question
    after a switch pays the load time.
    """
    result = Comparison(question=question)
    for model in models:
        try:
            result.answers.append(ask(index, question, model=model, **kwargs))
        except Exception as exc:  # a broken model must not lose the others
            log.exception("model %s failed", model)
            failed = Answer(question=question, model=model)
            failed.warnings.append(str(exc))
            result.answers.append(failed)
    return result
