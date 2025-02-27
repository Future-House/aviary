__all__ = [
    "LFRQAPairwiseEvalEnv",
    "LFRQAQuestion",
]

import logging
import random
import re
from collections.abc import Mapping
from typing import Any

from lmi import CommonLLMNames, LiteLLMModel, LLMModel
from paperqa.utils import strip_citations
from pydantic import Field, model_validator

from aviary.core import (
    Message,
    Messages,
    MultipleChoiceQuestion,
    ToolRequestMessage,
)
from aviary.envs.litqa import GradablePaperQAEnvironment

logger = logging.getLogger(__name__)

lfrqa_system_prompt = (
    # From RAG-QA Arena (https://arxiv.org/pdf/2407.13998) Table 13 and 15
    "Pairwise Evaluation: Instruction and Rubric\n"
    "We will show you a query and a pair of answers to the query. "
    "You need to provide your preference over this pair of answers.\n"
    "First, try your best to determine whether the information in an answer can help "
    "truthfully answer the query. Then rate your preference based on Helpfulness and "
    "Truthfulness.\n"
    "- Helpfulness: information that is helpful/relevant to answer the query. An ideal "
    "answer consists of only information that is helpful/relevant to answer the query.\n"
    "- Truthfulness: information that you believe is correct to answer the query. By our "
    "definition, truthful information should be helpful information. If you find it "
    "difficult to determine the truthfulness of some information, consider it untruthful. "
    "Often time, this is due to not enough context provided in the answer. Another source "
    "of untruthfulness is when conflicting information is presented, and the answer does "
    "not reconcile them in a coherent way.\n"
    "<rubric>\n"
    "Here is how you judge (in the order of importance):\n"
    "- If one answer has all truthful information while the other has some untruthful "
    "information, prefer the all-truthful one.\n"
    "- If both have some untruthful information, prefer the one with less untruthful "
    "information.\n"
    "- If both have all truthful information, prefer the one with more truthful or helpful "
    "information.\n"
    "- If two answers look equally good, or it is too hard to judge using the three cases "
    "above, then you are 'not sure' which one is better.\n"
    "</rubric>\n\n"
    "Label 1\n"
    "Query\n"
    "difference between 'publicly' and 'publically'.\n"
    "<query> difference between 'publicly' and 'publically'. </query>\n"
    "<answer 1> Both 'publicly' and 'publically' bear no difference in meaning, as they are "
    "essentially alternative spellings of the same concept. Publicly is more widely used, "
    "but the existence of 'publically' in reputable sources like the OED means it cannot be "
    "dismissed as simply incorrect. Some opinions hold that 'publicly' is the older irregular "
    "form, still preached by a lot of grammars, and 'publically,' on the other hand, is the "
    "newer and regular form. </answer 1>\n"
    "<answer 2> There is no difference in meaning between 'publicly' and 'publically'; they "
    "are alternative spellings of the same word. </answer 2>\n"
    "Both <answer 1> and <answer 2> are truthful. However, <answer 1> provides more truthful "
    "information as the context to compare the two terms. Therefore, <answer 1> is better.\n\n"
    "Label 2\n"
    "Query\n"
    "what did European/American historical cooks do with the egg whites?\n"
    "<query> what did European/American historical cooks do with the egg whites? </query>\n"
    "<answer 1> Historical European and American cooks used egg whites for making egg white "
    "omelettes and egg white pasta, as well as for stiffening clothing, similar to how starch "
    "is used today. Egg whites have found their place in various non-culinary applications "
    "throughout history, such as in the clarification of beer and wine, in the conservation of "
    "books through bookbinding and gilding, and in makeup as an ancient form of nail polish. "
    "</answer 1>\n"
    "<answer 2> They were also utilized historically as a stiffening agent for clothing, similar "
    "to how starch is used today. The culinary landscape was not left untouched, with egg whites "
    "making their way into recipes for omelettes and pastas as early as the 15th century. "
    "</answer 2>\n"
    "Both <answer 1> and <answer 2> provide several usages of egg whites for European/American "
    "cooks.\n"
    "<answer 2> clearly provides more options with fully explained details. Therefore, <answer 2> "
    "is better.\n\n"
    "Label 0\n"
    "Query\n"
    "should utf-16 be considered harmful?\n"
    "<query> should utf-16 be considered harmful? </query>\n"
    "<answer 1> The question of whether UTF-16 should be considered harmful is subject to differing "
    "opinions. One perspective suggests that UTF-16 is harmful due to a historical misconception "
    "about character encoding and recommends UTF-8 as the superior choice for various text interfaces. "
    "Another viewpoint argues that UTF-16 is not inherently harmful, emphasizing its utility in specific "
    "scenarios where it serves as a compromise between simplicity and compactness. The choice between "
    "UTF-16 and other encodings like UTF-8 depends on the specific requirements of the application, such "
    "as compatibility with ASCII or the need to efficiently encode certain character sets. </answer 1>\n"
    "<answer 2> UTF-16 should not be considered harmful. However, contrasting views argue that UTF-16 "
    "should indeed be considered harmful. Some argue that the very reason UTF-16 exists is because some "
    "time ago there used to be a misguided belief that WideChar is going to be what UCS-4 now is. "
    "Additionally, the harmfulness of UTF-16 is tied to issues with exercising code. </answer 2>\n"
    "Both <answer 1> and <answer 2> reconcile the two conflicting views with detailed explanation.\n"
    "I am not sure which one is better."
)

lfrqa_prompt_template = (
    # From RAG-QA Arena (https://arxiv.org/pdf/2407.13998) Table 14
    "Query is in the <query></query> tags. Answer 1 is in <answer 1></answer 1>,"
    "and Answer 2 is in <answer 2></answer 2>.\n"
    "<query> {question} </query>\n"
    "<answer 1> {answer1} </answer 1>\n"
    "<answer 2> {answer2} </answer 2>\n"
    "Review the rubric in <rubric> tags,\n"
    "- if you prefer <answer 1>, output 1.\n"
    "- if you prefer <answer 2>, output 2.\n"
    "- if you are not sure, output 0.\n"
    "First, think step by step, put your thinking in <thinking></thinking> tags.\n"
    "Your thinking must be shorter than 50 words.\n"
    "Then, provide your rating inside <rating></rating> tags.\n"
    "Remember your rating should be 0 if you are not sure, and your rating must be either 0, 1, or 2."
)


class LFRQAQuestion(MultipleChoiceQuestion):
    gt_doc_ids: list[int]
    grading_rewards: dict[str, float] = Field(
        default_factory=lambda: {"win": 1.0, "tie": 0.0, "lose": -1.0}
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_fields(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        processed_data = {
            "options": [],
            "prompt_without_options": True,
        }

        for k, v in data.items():
            if k == "answer":
                processed_data["ideal_answer"] = v
            elif k == "qid":
                processed_data["question_id"] = v
            elif k == "gold_doc_ids":
                processed_data["gt_doc_ids"] = v
            else:
                processed_data[k] = v

        if isinstance(processed_data["gt_doc_ids"], str):
            processed_data["gt_doc_ids"] = (
                processed_data["gt_doc_ids"].strip("[]").split(",")
            )
            processed_data["gt_doc_ids"] = [
                int(_id) for _id in processed_data["gt_doc_ids"]
            ]

        return processed_data

    def _extract_best_answer_index(self, text: str) -> int:
        match = re.search(r"<rating>(\d+)</rating>", text)
        return int(match.group(1)) if match else 0

    async def grade(  # type: ignore[override]
        self,
        proposed_answer: str,
        paper_search_ids: list[int],
        pairwise_eval_llm: LLMModel | str = CommonLLMNames.GPT_4O.value,
    ) -> dict[str, Any]:
        pqa_answer = strip_citations(proposed_answer)
        pqa_answer_index = 1 if random.random() < 0.5 else 2  # noqa: PLR2004
        data = {
            "question": self.question,
            "answer1": pqa_answer if pqa_answer_index == 1 else self.ideal_answer,
            "answer2": self.ideal_answer if pqa_answer_index == 1 else pqa_answer,
        }

        if isinstance(pairwise_eval_llm, str):
            pairwise_eval_llm = LiteLLMModel(name=pairwise_eval_llm)

        result = await pairwise_eval_llm.call_single(
            messages=[
                Message(role="system", content=lfrqa_system_prompt),
                Message(role="user", content=lfrqa_prompt_template.format(**data)),
            ],
        )

        best_answer_index = self._extract_best_answer_index(result.text or "")
        winner = (
            "paperqa"
            if best_answer_index == pqa_answer_index
            else "human"
            if best_answer_index != 0
            else "tie"
        )
        return {
            "evaluator_llm": pairwise_eval_llm.name,
            "qid": self.question_id,
            "question": self.question,
            "pqa_answer": pqa_answer,
            "human_answer": self.ideal_answer,
            "winner": winner,
            "paper_search_ids": paper_search_ids,
            "gt_doc_ids": self.gt_doc_ids,
            "pqa_answer_was_answer_1": pqa_answer_index == 1,
            "complete_evaluator_response": result.text,
        }


class LFRQAPairwiseEvalEnv(GradablePaperQAEnvironment[dict]):
    """Environment to evaluate paperqa's vs human's answers on Long Form RAG QA questions."""

    _query: LFRQAQuestion  # type: ignore[mutable-override]

    def __init__(
        self,
        query: LFRQAQuestion,
        *args,
        pairwise_eval_llm: LLMModel | str = CommonLLMNames.GPT_4O.value,
        **kwargs,
    ):
        kwargs["query"] = query
        super().__init__(*args, **kwargs)
        self.pairwise_eval_llm = pairwise_eval_llm

    async def _evaluate_answer(self) -> dict:
        evaluation = await self._query.grade(
            proposed_answer=self.state.session.answer,
            paper_search_ids=[
                int(doc.docname) for doc in self.state.docs.docs.values()
            ],
            pairwise_eval_llm=self.pairwise_eval_llm,
        )
        evaluation["llm"] = self._settings.llm
        reward = (
            self._rewards["win"]
            if evaluation["winner"] == "paperqa"
            else (
                self._rewards["lose"]
                if evaluation["winner"] == "human"
                else self._rewards["unsure"]
            )
        )
        evaluation["reward"] = reward

        return evaluation

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        messages, reward, done, truncated = await super(
            GradablePaperQAEnvironment, self
        ).step(action)
        if not done:
            return messages, reward, done, truncated
        evaluation = await self._evaluate_answer()
        if evaluation_callback := self._evaluation_callback:
            await evaluation_callback(evaluation)
        return messages, evaluation["reward"], done, truncated
