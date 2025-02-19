__all__ = [
    "LFRQAPairwiseEvalEnv",
    "LFRQAQuestion",
]

import logging
import random
import re
from uuid import UUID

from llmclient import CommonLLMNames, LiteLLMModel, LLMModel
from paperqa.docs import Docs
from paperqa.utils import strip_citations
from pydantic import BaseModel, model_validator

from aviary.core import (
    Messages,
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


class LFRQAPairwiseEvalEnv(GradablePaperQAEnvironment[dict]):
    """Environment to evaluate paperqa's vs human's answers on Long Form RAG QA questions."""

    def __init__(
        self,
        *args,
        qid: str | UUID,
        question: str,
        human_answer: str,
        gt_doc_ids: list[int],
        pairwise_eval_llm: LLMModel | str = CommonLLMNames.GPT_4O.value,
        **kwargs,
    ):
        kwargs["query"] = question
        kwargs["docs"] = Docs()
        super().__init__(*args, **kwargs)

        self.qid = qid
        self.question = question
        self.human_answer = human_answer
        self.gt_doc_ids = gt_doc_ids
        self.pairwise_eval_llm = pairwise_eval_llm

    def extract_best_answer_index(self, text: str) -> int:
        match = re.search(r"<rating>(\d+)</rating>", text)
        return int(match.group(1)) if match else 0

    async def _evaluate_answer(self) -> dict:
        """Pairwise evaluation of PaperQA vs Human answer."""
        paper_search_ids = [int(doc.docname) for doc in self.state.docs.docs.values()]

        pairwise_eval_llm = LiteLLMModel(name=self.pairwise_eval_llm)
        pqa_answer = strip_citations(self.state.session.answer)
        pqa_answer_index = 1 if random.random() < 0.5 else 2  # noqa: PLR2004
        data = {
            "question": self.question,
            "answer1": pqa_answer if pqa_answer_index == 1 else self.human_answer,
            "answer2": self.human_answer if pqa_answer_index == 1 else pqa_answer,
        }

        result = await pairwise_eval_llm.run_prompt(
            prompt=lfrqa_prompt_template,
            data=data,
            system_prompt=lfrqa_system_prompt,
        )

        best_answer_index = self.extract_best_answer_index(result.text)
        if best_answer_index == pqa_answer_index:
            winner, reward = "paperqa", self._rewards["win"]
        elif best_answer_index != 0:
            winner, reward = "human", self._rewards["lose"]
        else:
            winner, reward = "tie", self._rewards["tie"]

        return {
            "llm": self._settings.llm,
            "evaluator_llm": self.pairwise_eval_llm,
            "qid": self.qid,
            "question": self.question,
            "pqa_answer": pqa_answer,
            "human_answer": self.human_answer,
            "winner": winner,
            "paper_search_ids": paper_search_ids,
            "gt_doc_ids": self.gt_doc_ids,
            "pqa_answer_was_answer_1": pqa_answer_index == 1,
            "complete_evaluator_response": result.text,
            "reward": reward,
        }

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        messages, reward, done, truncated = await super().step(action)
        if not done:
            return messages, reward, done, truncated

        evaluation = await self._evaluate_answer()
        if evaluation_callback := self._evaluation_callback:
            await evaluation_callback(evaluation)

        return messages, evaluation["reward"], done, truncated


class LFRQAQuestion(BaseModel):
    qid: str | UUID
    question: str
    answer: str
    gt_doc_ids: list[int]

    @model_validator(mode="before")
    @classmethod
    def _validate_gt_doc_ids(cls, data: dict) -> dict:
        if data.get("gold_doc_ids") and not data.get("gt_doc_ids"):
            data["gt_doc_ids"] = data["gold_doc_ids"]
        if isinstance(data["gt_doc_ids"], str):
            data["gt_doc_ids"] = data["gt_doc_ids"].strip("[]").split(",")
            data["gt_doc_ids"] = [int(_id) for _id in data["gt_doc_ids"]]
        return data
