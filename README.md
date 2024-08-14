# QML Essentials

[![Pipx Status](https://servers.stroblme.de/api/badge/3/uptime/72?color=%2331c754&labelColor=%233f4850)](https://servers.stroblme.de/status/open) [![Lint and Pytest](https://github.com/cirKITers/qml-essentials/actions/workflows/python-app.yml/badge.svg)](https://github.com/cirKITers/qml-essentials/actions/workflows/python-app.yml) [![Page Build](https://github.com/cirKITers/qml-essentials/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/cirKITers/qml-essentials/actions/workflows/pages/pages-build-deployment)

This repo contains some of the commonly used Ansaetze and coding stuff required for working with QML and Data-Reuploading models.

## :rocket: Getting Started

You can find installation instructions and documentation on the corresponding [Github Page](https://cirkiters.github.io/qml-essentials/).

## :construction: Contributing

Contributions are very welcome!

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