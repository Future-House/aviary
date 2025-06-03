import json
import logging
import random
import re
from collections.abc import Collection
from enum import StrEnum, auto
from hashlib import md5
from pathlib import Path
from typing import ClassVar, cast

import pandas as pd
from aviary.core import TASK_DATASET_REGISTRY, MultipleChoiceQuestion, TaskDataset
from datasets import load_dataset
from ldp.alg import ComputeTrajectoryMetricsMixin

from .env import CloningEnv, CloningEnvTaskTypes
from .sequence_models import BioSequence, SequenceType
from .task_proteins import proteins

logger = logging.getLogger(__name__)


def convert_seqqa_to_task(
    question: str, answer: int | str | BioSequence, distractors: list[str]
) -> tuple[str, str | BioSequence, list[BioSequence]]:
    r"""Convert a question to a task with sequences.

    Examples:
        >>> question = "What is the percent GC of the DNA sequence GCA? Round to the nearest integer."
        >>> q, a, seqs = convert_seqqa_to_task(question, answer="48", distractors=['47' '49' '24']
        >>> q
        'What is the percent GC of the DNA sequence seq-32d9? Round to the nearest integer.\n\nOptions:\n48\n24\n47\n49'
        >>> a
        '48'
        >>> len(seqs)
        1
    """
    # Regular expression pattern to match words with 5 or more uppercase ACTG characters
    pattern = r"\b[ACTG]{5,}\b"

    # We pass answer too, since we don't know apriori which sequence(s) is in answer
    def convert_text(
        text: str, answer_: int | str | BioSequence
    ) -> tuple[str, list[BioSequence], str]:
        # sometimes the commas don't get correct spacing
        text = text.replace(",", ", ")
        answer_ = str(answer_)

        seqs = []
        for match in re.findall(pattern, text):
            name_hash = md5(match.encode()).hexdigest()[:4]  # noqa: S324
            s = BioSequence(
                sequence=match, type=SequenceType.DNA, name=f"seq-{name_hash}"
            )
            # only replace 1 - since we can have homologous sequences
            text = text.replace(match, f"{s.name}", 1)
            answer_ = answer_.replace(match, f"{s.name}", 1)
            seqs.append(s)

        return text, seqs, answer_

    question, seqs, answer = convert_text(question, answer)
    options = [answer, *distractors]
    options = random.sample(options, len(options))
    question += "\n\nOptions:\n"
    for option in options:
        # in the special case of mixed DNA/AA question, we DO NOT convert
        if question.startswith("What is the AA sequence of the longest ORF"):
            question += f"{option}\n"
            continue
        option, s, answer = convert_text(option, answer)
        seqs += s
        question += f"{option}\n"

    logger.debug(question)
    logger.debug(answer)
    return question, answer, seqs


class CloningDatasetSplit(StrEnum):
    small = auto()
    train = auto()
    val = auto()
    test = auto()
    all = auto()


class CloningSubDataset(StrEnum):
    SEQ_QA = "SeqQA"
    CLONING_SCENARIOS = "CloningScenarios"

    @property
    def small_split_size(self) -> int:
        return 100 if self == CloningSubDataset.SEQ_QA else 10


class CloningDataset(TaskDataset[CloningEnv], ComputeTrajectoryMetricsMixin):
    """A dataset of cloning tasks - use split = small or split = train for complete.

    The dataset loads SeqQA or CloningScenarios datasets from Lab-Bench.
    """

    Split: ClassVar = CloningDatasetSplit

    def __init__(
        self,
        split: CloningDatasetSplit = CloningDatasetSplit.train,
        subset_name: str | CloningSubDataset = CloningSubDataset.SEQ_QA,
        subtasks: Collection[str] | None = None,
        data_source: tuple[str, str] | str | None = None,
        collapse_prior_user_message: bool = False,
        include_cannot_answer: bool = False,
        include_convention: bool = False,
        shuffle: bool = False,
    ):
        """Initialize.

        Args:
            split: Train or small split.
            subset_name: Task subset. Defaults to SeqQA
            subtasks: An optional set of subtasks to filter by.
            data_source: Optional data source for this dataset. It can be:
                - `None`: hardcode LAB-Bench public Hugging Face dataset, hardcode split to "train".
                  TODO: this is not a good default, since it does not contain the fixed questions.
                - `str`: custom Hugging Face dataset or local directory path to SeqQA
                  (e.g. `path/to/LAB-Bench-internal/SeqQA`), hardcode split to "train".
                - `tuple[str, str]`: value 1 is a custom Hugging Face dataset,
                  value 2 is a custom split.
            collapse_prior_user_message: Append user message into the prior messages when creating
                a ToolRequestMessage. Need this for for Mistral models. Defaults to False.
            include_cannot_answer: Include "Cannot answer" in the distractors. Defaults to False.
            include_convention: Include ambiguous conventions in questions
            shuffle: Legacy, do not use.
        """
        if isinstance(subset_name, CloningSubDataset):
            subset = subset_name
        else:
            try:
                subset = CloningSubDataset(subset_name)
            except KeyError:
                raise ValueError(
                    f"Invalid subset: {subset_name}. Only SeqQA and CloningScenarios allowed."
                ) from None

        self.dataset = self.load_dataset(data_source, subset)
        self.collapse_prior_user_message = collapse_prior_user_message
        self.include_cannot_answer = include_cannot_answer

        # now we need to filter out:
        # questions w RNA
        self.dataset = self.dataset[
            self.dataset.apply(lambda x: "RNA" not in x["question"], axis=1)
        ]
        if include_convention and subset == CloningSubDataset.SEQ_QA:
            self.dataset["question"] = self.dataset["question"].apply(
                lambda x: x
                + "\nNote: Only consider coding ORFs and assume position indices start from 1."
            )
        # for backward compatibility
        if shuffle:
            logger.warning(
                "Shuffle flag was dropped from CloningDataset - it was shuffled implicitly by sorting on UUID4"
            )
        self.dataset = self.dataset.reset_index(drop=True)

        # HACK to define splits

        match split:
            case CloningDatasetSplit.small:
                self.dataset = self.dataset[: subset.small_split_size]

            case CloningDatasetSplit.train:
                d = self.dataset[self.dataset["split"] == "public"]
                num = len(d)
                self.dataset = d[: int(num * 0.9)]

            case CloningDatasetSplit.val:
                d = self.dataset[self.dataset["split"] == "public"]
                num = len(d)
                self.dataset = d[int(num * 0.9) :]

            case CloningDatasetSplit.test:
                d = self.dataset[self.dataset["split"] == "private"]
                if len(d) == 0:
                    raise ValueError("No private tasks found in the dataset.")
                self.dataset = d

            case CloningDatasetSplit.all:
                pass

            case _:
                raise ValueError(f"Invalid split: {split}")

        if subtasks is not None:
            self.dataset = self.dataset[
                self.dataset.apply(lambda x: x["subtask"] in subtasks, axis=1)
            ]

        if not len(self.dataset):
            raise RuntimeError(
                "No tasks found in the dataset - filters may have been too stringent."
            )

    def load_dataset(
        self, data_source: tuple[str, str] | str | None, subset: CloningSubDataset
    ) -> pd.DataFrame:
        """Load the dataset, assign split, and sort by id."""
        if not data_source:
            data_source = "futurehouse/lab-bench"
            split = "train"
        elif isinstance(data_source, str):
            split = "train"
        else:
            data_source, split = data_source
            if Path(data_source).exists():
                raise ValueError(
                    "When specifying a data_source to include a custom split, only"
                    " Hugging Face-hosted datasets are supported, but a local directory"
                    f" {data_source!r} was passed. The reason is datasets in local"
                    " directories don't have a concept of a split."
                )

        if not Path(data_source).exists():
            # Assume it's a Huggingface dataset if we can't find it locally
            data = load_dataset(data_source, name=subset, split=split).to_pandas()
            data["split"] = "public"
            return data.sort_values(by="id")

        # Crawl the directory for task JSONLs.
        dfs: list[pd.DataFrame] = []
        for jsonl in Path(data_source).rglob("*.jsonl"):
            d = pd.read_json(jsonl, lines=True)
            d["task"] = [jsonl.name.split(".jsonl")[0]] * len(d)
            dfs.append(d)
        data = pd.concat(dfs)

        # load splits
        private_ids = []
        for p in Path(data_source).rglob("*.json"):
            with p.open() as f:
                split_data = json.load(f)
            private_ids.extend(split_data["private"])
        data["split"] = data.id.apply(
            lambda x: "private" if (x in private_ids) else "public"
        )
        # we may have duplicates if we loaded the full and public subsets
        # sort so we are not affected by glob order
        return data.drop_duplicates(subset=["id"], keep="first").sort_values(by="id")

    def __len__(self) -> int:
        return len(self.dataset)

    # TODO: move to MultipleChoiceQuestion.DEFAULT_UNSURE_OPTION
    CANNOT_ANSWER: ClassVar[str] = "Cannot answer"

    def get_new_env_by_idx(self, idx: int) -> CloningEnv:
        row = self.dataset.iloc[idx]
        question, distractors, ideal = row["question"], row["distractors"], row["ideal"]
        mcq_kwargs = {
            "question": question,
            "options": tuple(distractors),  # Effectively a deepcopy
            "ideal_answer": str(ideal),
            "shuffle_seed": "SEED_USING_QUESTION",
        }
        if self.include_cannot_answer:
            mcq = MultipleChoiceQuestion(
                **mcq_kwargs, unsure_answer=type(self).CANNOT_ANSWER
            )
            distractors.append(type(self).CANNOT_ANSWER)
        else:
            mcq = MultipleChoiceQuestion(**mcq_kwargs)
        problem, answer, seqs = convert_seqqa_to_task(question, ideal, distractors)
        # TODO: handle that mcq.question_prompt and problem have different options ordering

        logger.debug(problem, answer, seqs)
        return CloningEnv(
            problem_id=row["id"],
            problem=problem,
            start=seqs,
            answer=answer,
            collapse_prior_user_message=self.collapse_prior_user_message,
            task_type=CloningEnvTaskTypes.SEQUENCE
            if isinstance(answer, BioSequence)
            else CloningEnvTaskTypes.DIRECT,
            mcq=mcq,
        )

    def get_new_env(self) -> CloningEnv:
        i = random.randint(0, len(self) - 1)
        return self.get_new_env_by_idx(i)


class HardCloningDataset(TaskDataset[CloningEnv], ComputeTrajectoryMetricsMixin):
    def __init__(self, split: str = "train"):
        tasks = [
            {
                "prompt": "Clone the given protein {protein} into {plasmid} (check annotations near MCS are inside lacZ)",
                "plasmids": ["pUC18", "pUC19"],
                "proteins": ["lacI", "galA", "HIS3", "LEU2", "UBB", "HBB"],
            },
            {
                "prompt": "Clone the given protein {protein} into {plasmid} to express in E. coli (check annotations downstream of an e. coli promoter, check for e coli origin, check for e coli selection)",
                "plasmids": ["mEGFP-pBAD", "pTDpelB-C_sfYFPTwinStrep"],
                "proteins": ["lacI", "galA"],
            },
            {
                "prompt": "Clone the given protein {protein} into {plasmid} to express in human cells (check annotations downstream of a mammalian promoter, check for human origin, check for human selection)",
                "plasmids": ["sfGFP-N1", "pmEGFP-1"],
                "proteins": ["UBB", "HBB"],
            },
            {
                "prompt": "Clone the given protein {protein} into {plasmid} to express in yeast with a GFP fusion (check annotations above, plus GFP in correct relative orientation)",
                "plasmids": ["pRS415-GPD", "pKT0128"],
                "proteins": ["UBB", "HBB"],
            },
        ]
        random.seed(0)
        examples = []
        for t in tasks:
            for pl in t["plasmids"]:
                for pro in t["proteins"]:
                    logger.debug(proteins[pro].strip().replace("\n", ""))
                    examples.append({
                        "task": cast(str, t["prompt"]).format(protein=pro, plasmid=pl),
                        "start": BioSequence(
                            sequence=proteins[pro].strip().replace("\n", ""),
                            is_circular=False,
                            type="dna",
                            name=pro,
                        ),
                    })

        self.dataset = examples

    def __len__(self) -> int:
        return len(self.dataset)

    def get_new_env_by_idx(self, idx: int) -> CloningEnv:
        example = self.dataset[idx]
        return CloningEnv(
            problem_id=str(idx),
            problem=str(example["task"]).split("(")[0],
            start=[cast(BioSequence, example["start"])],
            answer=str(example["task"]).split("(")[1].split(")")[0],
            task_type=CloningEnvTaskTypes.PLASMID,
        )

    def get_new_env(self) -> CloningEnv:
        i = random.randint(0, len(self) - 1)
        return self.get_new_env_by_idx(i)


TASK_DATASET_REGISTRY["cloning"] = "cloning.dataset", "CloningDataset"
