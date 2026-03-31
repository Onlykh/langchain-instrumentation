# Contributing

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Run checks

```bash
pytest -q
python -m build
python -m twine check dist/*
```

## Pull requests

- Keep PRs focused and small.
- Add or update tests for behavior changes.
- Update `README.md` when user-facing behavior changes.
- Update `CHANGELOG.md` for release-facing changes.
