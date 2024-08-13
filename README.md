# QML Essentials

This repo contains some of the commonly used Ansaetze and coding stuff required for working with QML and Data-Reuploading models.

## Installation

![Pipx Status](https://servers.stroblme.de/api/badge/3/status)    
[![Python application](https://github.com/cirKITers/qml-essentials/actions/workflows/python-app.yml/badge.svg)](https://github.com/cirKITers/qml-essentials/actions/workflows/python-app.yml)

The package is available at [this index](https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org/lc3267/quantum).

Assuming you have Poetry installed
- `poetry source add --priority=supplemental quantum https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org/lc3267/quantum/+simple/`
- `poetry add --source quantum qml-essentials`

With plain pip:
- `pip install --index-url https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org/lc3267/quantum/+simple/ qml-essentials`

## Contributing

Building and packaging requires some extra steps (assuming Poetry):
- `poetry run devpi use https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org`
- `poetry run devpi login alice --password=456`
- `poetry run devpi use alice/quantum`
- `poetry config repositories.quantum https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org/lc3267/quantum`
- `poetry config http-basic.quantum alice 456` (or remove password for interactive prompt)
- `poetry version (major|minor|patch|premajor|preminor|prepatch)` as explained [here](https://python-poetry.org/docs/cli/#version)
- `poetry publish --build -r quantum`
