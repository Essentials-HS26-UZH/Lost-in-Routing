# Lost-in-Routing
Multilingual LLM query routing: evaluating language-aware routers for cost-efficient small-vs-large model selection.

## Setup

```bash
conda env create -f environment.yml
conda activate router
```
To deactivate an active environment, use:

```bash
conda deactivate
```
To update the environment:

```bash
conda env update -f environment.yml --prune
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
