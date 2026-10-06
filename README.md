# Lost-in-Routing
Multilingual LLM query routing: evaluating language-aware routers for cost-efficient small-vs-large model selection.

## Prerequisite

Create the Conda environment *once*:
```bash
conda env create -f environment.yml
```

To update the environment, use:
```bash
conda env update -f environment.yml --prune
```

## Setup

Activate the environment:
```bash
conda activate router
```
To deactivate an active environment, use:

```bash
conda deactivate
```

## Workflow

1. Create a branch (e.g. feature/short-description, fix/short-description)

2. Write code + tests

3. Run checks:

```bash
pytest
ruff check .
mypy src/
```

4. Commit (e.g. feat(context): short descrption, fix(context): short description)

5. Push your branch and create a Pull Request.


## Project structure

```text
docs/       # Documentation
notebooks/  # Experiments and analysis
src/        # Project code
tests/      # Unit tests
```
