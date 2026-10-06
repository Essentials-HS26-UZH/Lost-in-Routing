# Development

## Development Environment

We use **Conda** to manage the project's Python environment. The environment is defined in `environment.yml`, which is committed to the repository so that all team members can work with the same dependencies and Python version.

Conda was chosen because the project uses machine learning and scientific computing libraries, which can have complex dependencies and may require different configurations depending on the hardware.

## Code Quality

We use **Ruff** for linting and formatting and **mypy** for type checking.

Ruff was chosen because it provides the main code-quality checks we need in a single tool. Mypy is used alongside Python type hints to catch type-related errors before they reach runtime.

## Testing

We use **pytest** for unit testing. Tests are kept separately from the project code in the `tests/` directory.

Testing is particularly important for this project because the router consists of several components, such as data processing, language detection, and routing logic, which can be tested independently.

## Dependencies

Where possible, we use established libraries rather than implementing functionality ourselves. This is especially relevant for machine learning, data processing, and language-related functionality.

Using existing, well-maintained libraries reduces unnecessary implementation effort and makes the project easier to maintain and reproduce.

## Git Workflow

The `main` branch is kept stable and is protected from direct pushes. Development is done in separate branches and integrated through Pull Requests.

This gives us a simple review point before changes become part of the main project and allows automated tests and code-quality checks to run before merging.
