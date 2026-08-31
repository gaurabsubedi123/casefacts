# Shortcuts. Everything here also works as a plain command — see README.md.

VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: install ui test doctor models clean

install:            ## create the venv and install the tool
	uv venv || python3 -m venv $(VENV)
	uv pip install -e ".[dev]" || $(PY) -m pip install -e ".[dev]"

ui:                 ## start the local web interface on http://127.0.0.1:5002
	$(PY) -m casefacts.cli ui

test:               ## run the test suite (no model needed)
	$(PY) -m pytest -q

doctor:             ## check Ollama, the models, ocrtool and the index
	$(PY) -m casefacts.cli doctor

models:             ## what is installed, and what each one is good for
	$(PY) -m casefacts.cli models

clean:              ## remove caches (never touches the index or your documents)
	rm -rf .pytest_cache **/__pycache__ casefacts/__pycache__ casefacts/web/__pycache__ tests/__pycache__
