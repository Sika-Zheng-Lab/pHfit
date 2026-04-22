# Contributing to pHfit

Thank you for your interest in contributing to pHfit!

## Development setup

```bash
git clone https://github.com/Sika-Zheng-Lab/pHfit.git
cd pHfit
pip install -e ".[dev]"
```

Supported Python versions: 3.10 – 3.14.

## Running tests

```bash
pytest tests/ -v --cov=phfit
```

All pull requests are tested via GitHub Actions across the supported Python matrix.

## Branching and release workflow

- `main` — released, stable code. Protected.
- `develop` — integration branch for upcoming work.
- Feature branches: branch off `develop`, open PR back to `develop`.

### Releasing a new version

Releases are automated by GitHub Actions and are triggered by **merging a pull request from `develop` into `main`**:

1. On `develop`, bump `version` in [pyproject.toml](pyproject.toml) and update [CHANGELOG.md](CHANGELOG.md) with the new version and date. Update `version` and `date-released` in [CITATION.cff](CITATION.cff).
2. Open a PR from `develop` into `main`. Use the changelog entry as the PR description — it becomes the GitHub Release body.
3. After review, merge the PR. The following will run automatically:
   - [`release.yml`](.github/workflows/release.yml) — creates the `vX.Y.Z` git tag, GitHub Release, and pushes Docker images (`vX.Y.Z` and `latest`) to Docker Hub.
   - [`publish.yml`](.github/workflows/publish.yml) — builds the sdist + wheel and uploads to PyPI (uses `--skip-existing` so re-runs are safe).
   - [Zenodo](https://zenodo.org) (if enabled) — mints a DOI for the release.

## Pull request checklist

- [ ] Tests added or updated for new behavior.
- [ ] `pytest tests/ -v` passes locally.
- [ ] [CHANGELOG.md](CHANGELOG.md) updated for user-visible changes.
- [ ] Documentation (README, docstrings) updated where relevant.

## Reporting issues

Please use the [issue templates](https://github.com/Sika-Zheng-Lab/pHfit/issues/new/choose) for bug reports, feature requests, and questions. For security issues, see [SECURITY.md](SECURITY.md).
