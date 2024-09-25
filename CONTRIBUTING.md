# Contributing to QUAFEL

:tada: Welcome! :tada:

Contributions are highly welcome!
Start of by..
1. Creating an issue using one of the templates (Bug Report, Feature Request)
   - let's discuss what's going wrong or what should be added
   - can you contribute with code? Great! Go ahead! :rocket:
2. Forking the repository and working on your stuff
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

## Testing

We do our testing with Pytest. Corresponding tests can be triggered as follows:
```
poetry run pytest
```

## Packaging

Building and packaging requires some extra steps (assuming Poetry):
- `poetry run devpi use https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org`
- `poetry run devpi login alice --password=456`
- `poetry run devpi use alice/quantum`
- `poetry config repositories.quantum https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org/lc3267/quantum`
- `poetry config http-basic.quantum alice 456` (or remove password for interactive prompt)
- `poetry version (major|minor|patch|premajor|preminor|prepatch)` as explained [here](https://python-poetry.org/docs/cli/#version)
- `poetry publish --build -r quantum`

## Re-Installing Package

If you want to overwrite the latest build you can simply push to the package index with the recent changes.
Updating from this index however is then a bit tricky, because poetry keeps a cache of package metadata.
To overwrite an already installed package with the same version (but different content) follow these steps:
1. `poetry remove qml_essentials`
2. `poetry cache clear quantum --all`
3. `poetry add qml_essentials@latest` (or a specific version)

Note that this will also re-evaluate parts of other dependencies, and thus may change the `poetry.lock` file significantly.

## Documentation

We use MkDocs for our documentation. To run a server locally, run:
```
poetry run mkdocs serve
```
This will automatically trigger a rebuild each time you make changes.
See the [MkDocs Documentation](https://cirkiters.github.io/qml-essentials/usage/) for more details.

For pushing to Github pages:
```
poetry run mkdocs gh-deploy
```