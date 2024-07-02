# QML Essentials

This repo contains some of the commonly used Ansaetze and coding stuff required for working with QML and Data-Reuploading models.

## Installation

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
- `poetry config repositories.foo https://ea3a0fbb-599f-4d83-86f1-0e71abe27513.ka.bw-cloud-instance.org/lc3267/quantum`
- `poetry config http-basic.foo alice 456` (or remove password for interactive prompt)
- `poetry publish --build -r foo`