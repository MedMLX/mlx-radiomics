PYTHON ?= .venv/bin/python
TESTS ?= tests
PYTEST_ARGS ?=

.PHONY: env qa test docs

env:
	uv sync --locked --group dev

qa:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pyright --pythonpath "$(PYTHON)"

test:
	$(PYTHON) -m pytest $(TESTS) $(PYTEST_ARGS)

# Read the Docs builds the same pinned requirements; uv sync removes them again.
docs:
	uv pip install --python $(PYTHON) -q -r docs/requirements.txt
	$(PYTHON) -m sphinx -W --keep-going -c docs/reference -b html docs docs/_build/html
