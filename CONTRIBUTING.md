# Contributing to QML-Essentials

:tada: Welcome! :tada:

Contributions are highly welcome!
Start of by..
1. Creating an issue using one of the templates (Bug Report, Feature Request)
   - let's discuss what's going wrong or what should be added
   - can you contribute with code? Great! Go ahead! :rocket:
2. Forking the repository and working on your stuff. See the sections below for details on how to set things up.
3. Creating a pull request to the main repository

## Setup

Contributing to this project requires some more dependencies besides the "standard" packages.
Those are specified in the groups `dev` and `docs`.
```
poetry install --with dev,docs
```

Additionally, we have pre-commit hooks in place, which can be installed as follows: 
```
poetry run pre-commit autoupdate
poetry run pre-commit install
```

Currently the only purpose of the hook is to run Black on commit which will do some code formatting for you.
However be aware, that this might reject your commit and you have to re-do the commit.

## Testing

We do our testing with Pytest. Corresponding tests can be triggered as follows:
```
poetry run pytest
```
There are Github action pipelines in place, that will do linting and testing once you open a pull request.
However, it's a good idea to run tests and linting (either Black or Flake8) locally before pushing.

## Packaging

Packaging is done automagically using Github actions.
This action is triggered when a new version is being detected in the `pyprojec.toml` file.
This works by comparing the output of `poetry version --short` against `git tag` and triggering the publishing process if those differ.
Publishing includes
- setting the git tag equal to the version specified in `pyproject.toml`
- creating a release with the current git tag and automatically generated release notes
- publishing the package to PyPI using the stored credentials

## Documentation

We use MkDocs for our documentation. To run a server locally, run:
```
poetry run mkdocs serve
```
This will automatically trigger a rebuild each time you make changes.
See the [MkDocs Documentation](https://cirkiters.github.io/qml-essentials/usage/) for more details.

Publishing (and building) the documentation is done automagically using Github actions.
This action is triggered when a new release is made.