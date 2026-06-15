# Contributing

Thanks for considering a contribution to `veilrouter`.

## Development setup

```powershell
python -m pip install -e ".[dev,foundry]"
python -m pytest -q
```

The Foundry Local extra is only required for live local-model tests or scorer
benchmarking. Unit tests use fakes and should not require model downloads.

## Pull request expectations

- Keep changes focused and documented.
- Add or update tests for behavior changes.
- Do not commit model weights, credentials, API keys, or generated caches.
- Run `python -m pytest -q` before opening a pull request.

## Release process

Releases are expected to be created from a clean `main` branch after tests pass.
Publishing should use the repository's GitHub Actions PyPI workflow with PyPI
Trusted Publishing configured for this repository.
