import os

from aviary.envs.lfrqa.env import LFRQAQuestion, LFRQAPairwiseEvalEnv
from aviary.envs.lfrqa.task import LFRQATaskDataset
import pandas as pd

def test_env_construction() -> None:
    data: list[LFRQAQuestion] = [
        LFRQAQuestion(**row)
        for row in pd.read_csv(
            os.path.join("packages", "lfrqa", "tests", "datasets", "mini_lfrqa.csv")
        )[["qid", "question", "answer", "gold_doc_ids"]].to_dict(orient="records")
    ]
    
    dataset = LFRQATaskDataset(
        data=data,
    )

    env = dataset.get_new_env_by_idx(0)
    assert isinstance(env, LFRQAPairwiseEvalEnv)
