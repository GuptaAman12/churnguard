# Contributing

Thanks for taking a look. This repository is primarily a reference
implementation, but the workflow below is the same one used in production teams.

## Setup

```bash
uv sync --all-groups
uv run pre-commit install
uv run dvc repro
```

## Before opening a PR

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest
```

All four must pass. CI runs the same commands plus Docker builds, Terraform
validation, Helm linting and a security scan.

## Conventions

- **No notebook-only logic.** Anything that affects predictions belongs in
  `src/churnguard/` and must be covered by a test.
- **Schemas are contracts.** Changing `schema.py` or `constants.py` requires a
  migration note and, if the feature set changes, a new model version.
- **Config is typed.** Add new knobs to `Settings` with a safe default; never
  read `os.environ` directly outside `config.py`.
- **Logs are structured.** Use `get_logger(__name__)` and keyword arguments.
- **Commits** follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
- **Promotion logic** must never be weakened to make a demo pass. If a test
  needs a lower bar, override `min_roc_auc` in the test fixture.

## Adding a dependency

```bash
uv add <package>            # runtime
uv add --group dev <pkg>    # tooling
```

Commit both `pyproject.toml` and `uv.lock`. If the library ships no type
information, add it to the `mypy` overrides rather than sprinkling
`# type: ignore`.

## Reporting a security issue

Do not open a public issue. Email the maintainers listed in `pyproject.toml`
with reproduction steps and affected versions.
