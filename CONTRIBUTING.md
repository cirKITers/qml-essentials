### Packaging

Building and packaging requires some extra steps (assuming Poetry):
- `poetry run devpi use https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org`
- `poetry run devpi login alice --password=456`
- `poetry run devpi use alice/quantum`
- `poetry config repositories.quantum https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org/lc3267/quantum`
- `poetry config http-basic.quantum alice 456` (or remove password for interactive prompt)
- `poetry version (major|minor|patch|premajor|preminor|prepatch)` as explained [here](https://python-poetry.org/docs/cli/#version)
- `poetry publish --build -r quantum`

### Documentation

For local testing:
- `mkdocs serve`

For pushing to Github pages:
- `mkdocs gh-deploy`