import asyncio

import click
from ldp.agent import ReActAgent

from .env import CloningEnv


async def amain(problem: str) -> None:
    env = CloningEnv(problem=problem)
    agent = ReActAgent(llm_model={"name": "openai/gpt-4o-2024-08-06"})
    obs, tools = await env.reset()
    state = await agent.init_state(tools)
    for _ in range(8):
        action, state, _ = await agent.get_asv(state, obs)
        obs, _, done, _ = await env.step(action.value)
        print(obs[-1].content)
        frame = env.export_frame()
        print("*" * 80)
        print(frame.state)
        if done:
            break


@click.command()
@click.argument("problem", default="Design a plasmid that expresses GFP in E. coli.")
def main(problem: str) -> None:
    asyncio.run(amain(problem))


if __name__ == "__main__":
    main()
