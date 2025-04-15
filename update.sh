# Convenience script to update all major releases

poetry add pennylane@latest
poetry add dill@latest

poetry add black@latest --group dev
poetry add pytest@latest --group dev
poetry add flake8@latest --group dev
poetry add pre-commit@latest --group dev
poetry add coverage@latest --group dev
poetry add licensecheck@latest --group dev
poetry add matplotlib@latest --group dev
poetry add pytest-xdist@latest --group dev

poetry add mkdocs@latest --group docs
poetry add mkdocs-material@latest --group docs
poetry add mkdocs-glightbox@latest --group docs
poetry add mkdocstrings-python@latest --group docs
poetry add markdown-include@latest --group docs
poetry add ipykernel@latest --group docs

poetry update