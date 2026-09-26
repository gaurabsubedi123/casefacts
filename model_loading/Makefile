VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: install run test exe clean

install:            ## create the venv and install the portal
	uv venv || python3 -m venv $(VENV)
	uv pip install -e ".[dev]" || $(PY) -m pip install -e ".[dev]"

run:                ## start the portal and open the browser
	$(PY) -m modelportal

test:               ## run the test suite (no model or network needed)
	$(PY) -m pytest -q

exe:                ## build a single-file executable for THIS operating system
	uv pip install -e ".[exe]" || $(PY) -m pip install -e ".[exe]"
	$(VENV)/bin/pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/ModelPortal.spec

clean:
	rm -rf build dist
