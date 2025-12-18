# Contributing to QML-Essentials

Contributions are highly welcome! :hugging_face:

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
uv install --with dev,docs
```

Additionally, we have pre-commit hooks in place, which can be installed as follows: 
```
uv run pre-commit autoupdate
uv run pre-commit install
```

Currently the only purpose of the hook is to run Black on commit which will do some code formatting for you.
However be aware, that this might reject your commit and you have to re-do the commit.

## Testing

We do our testing with Pytest.
There are Github action pipelines in place, that will do linting and testing once you open a pull request.
However, it's a good idea to run tests and linting (either Black or Flake8) locally before pushing, e.g.
```
uv run black .
uv run pytest --dist load -m "not expensive" -n auto
```
Which will run all tests that are not marked as expensive.
See [Pytest](https://pytest.org/) for more details on how to run specific tests only.

## Packaging

Packaging is done automagically using Github actions.
This action is triggered when a new version is being detected in the `pyprojec.toml` file.
This works by comparing the output of `uv version --short` against `git tag` and triggering the publishing process if those differ.
Publishing includes
- setting the git tag equal to the version specified in `pyproject.toml`
- creating a release with the current git tag and automatically generated release notes
- publishing the package to PyPI using the stored credentials

## Documentation

We use MkDocs for our documentation. To run a server locally, run:
```
uv run mkdocs serve
```
This will automatically trigger a rebuild each time you make changes.
See the [MkDocs Documentation](https://cirkiters.github.io/qml-essentials/usage/) for more details.

Publishing (and building) the documentation is done automagically using Github actions.
This action is triggered when a new release is made.