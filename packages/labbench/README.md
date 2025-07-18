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
  subclass for QA where a single-image is pre-added.

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

## References

[1] Skarlinski et al.
[Language agents achieve superhuman synthesis of scientific knowledge](https://arxiv.org/abs/2409.13740).
ArXiv:2409.13740, 2024.

[2] Laurent et al.
[LAB-Bench: Measuring Capabilities of Language Models for Biology Research](https://arxiv.org/abs/2407.10362).
ArXiv:2407.10362, 2024.
