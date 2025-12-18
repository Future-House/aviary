# aviary.labbench

LAB-Bench environments implemented with aviary,
allowing agents to perform question answering on scientific tasks.

## Installation

To install the LAB-Bench environment, run:

```bash
pip install 'fhaviary[labbench]'
```

## Usage

In [`labbench/env.py`](src/aviary/envs/labbench/env.py), you will find:

- `GradablePaperQAEnvironment`: an PaperQA-backed environment
  that can grade answers given an evaluation function.
- `ImageQAEnvironment`: an `GradablePaperQAEnvironment`
  subclass for QA where image(s) are pre-added.

And in [`labbench/task.py`](src/aviary/envs/labbench/task.py), you will find:

- `TextQATaskDataset`: a task dataset designed to
  pull down FigQA, LitQA2, or TableQA from Hugging Face,
  and create one `GradablePaperQAEnvironment` per question.
- `ImageQATaskDataset`: a task dataset that pairs with `ImageQAEnvironment`
  for FigQA or TableQA.

Here is an example of how to use them:

```python
import os

from ldp.agent import SimpleAgent
from ldp.alg import Evaluator, EvaluatorConfig, MeanMetricsCallback
from paperqa import Settings

from aviary.env import TaskDataset


async def evaluate(folder_of_litqa_v2_papers: str | os.PathLike) -> None:
    settings = Settings(paper_directory=folder_of_litqa_v2_papers)
    dataset = TaskDataset.from_name("litqa2", settings=settings)
    metrics_callback = MeanMetricsCallback(eval_dataset=dataset)

    evaluator = Evaluator(
        config=EvaluatorConfig(batch_size=3),
        agent=SimpleAgent(),
        dataset=dataset,
        callbacks=[metrics_callback],
    )
    await evaluator.evaluate()
    print(metrics_callback.eval_means)
```

### Image Question-Answer

This is an environment/dataset for giving PaperQA a `Docs` object with
the image(s) for one LAB-Bench question.
It's designed to be a comparison with zero-shotting the question to a LLM,
but instead of a singular prompt the image is put through the PaperQA agent loop.

```python
from typing import cast

import litellm
import pytest
from ldp.agent import Agent
from ldp.alg import (
    Evaluator,
    EvaluatorConfig,
    MeanMetricsCallback,
    StoreTrajectoriesCallback,
)
from paperqa.settings import AgentSettings, IndexSettings

from aviary.envs.labbench import (
    ImageQAEnvironment,
    ImageQATaskDataset,
    LABBenchDatasets,
)


@pytest.mark.asyncio
async def test_image_qa(tmp_path) -> None:
    litellm.num_retries = 8  # Mitigate connection-related failures
    settings = ImageQAEnvironment.make_base_settings()
    settings.agent = AgentSettings(
        agent_type="ldp.agent.SimpleAgent",
        index=IndexSettings(paper_directory=tmp_path),
        # TODO: add image support for paper_search
        tool_names={"gather_evidence", "gen_answer", "complete", "reset"},
        agent_evidence_n=3,  # Bumped up to collect several perspectives
    )
    dataset = ImageQATaskDataset(dataset=LABBenchDatasets.TABLE_QA, settings=settings)
    t_cb = StoreTrajectoriesCallback()
    m_cb = MeanMetricsCallback(eval_dataset=dataset, track_tool_usage=True)
    evaluator = Evaluator(
        config=EvaluatorConfig(
            batch_size=256,  # Use batch size greater than FigQA size and TableQA size
            max_rollout_steps=18,  # Match aviary paper's PaperQA setting
        ),
        agent=cast(Agent, await settings.make_ldp_agent(settings.agent.agent_type)),
        dataset=dataset,
        callbacks=[t_cb, m_cb],
    )
    await evaluator.evaluate()
    print(m_cb.eval_means)
```

## References

[1] Skarlinski et al.
[Language agents achieve superhuman synthesis of scientific knowledge](https://arxiv.org/abs/2409.13740).
ArXiv:2409.13740, 2024.

[2] Laurent et al.
[LAB-Bench: Measuring Capabilities of Language Models for Biology Research](https://arxiv.org/abs/2407.10362).
ArXiv:2407.10362, 2024.
