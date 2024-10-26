# Contributing to aviary

## Repo Structure

aviary is a monorepo using
[`uv`'s workspace layout](https://docs.astral.sh/uv/concepts/workspaces/#workspace-layouts).

## Installation

1. Git clone this repo
2. Install the project manager `uv`:
   https://docs.astral.sh/uv/getting-started/installation/
3. Run `uv sync`

This will editably install the full monorepo in your local environment.

## Testing

To run tests, please just run `pytest` in the repo root.

Note you will need OpenAI and Anthropic API keys configured.
