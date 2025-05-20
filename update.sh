# Convenience script to update all major releases of explicit dependencies

# all main dependencies
poetry add pennylane@latest
poetry add dill@latest

# all dev dependencies
poetry add black@latest --group dev
poetry add pytest@latest --group dev
poetry add flake8@latest --group dev
poetry add pre-commit@latest --group dev
poetry add licensecheck@latest --group dev
poetry add matplotlib@latest --group dev
poetry add pytest-xdist@latest --group dev
poetry add coverage@latest --group dev

# all docs dependencies
poetry add mkdocs@latest --group docs
poetry add mkdocs-material@latest --group docs
poetry add mkdocs-glightbox@latest --group docs
poetry add mkdocstrings-python@latest --group docs
poetry add markdown-include@latest --group docs
poetry add ipykernel@latest --group docs

# finally update minor versions of implicit dependencies
poetry update