# fullwave2-ultra (public package) — Python dev tasks.
# The CUDA solver binaries are built & distributed SEPARATELY (private) and fetched
# at runtime; there is no CUDA build here. See README.md "Running the solver".
.PHONY: install install-dev test lint fixtures leak-check clean

install:
	pip install -e .
install-dev:
	pip install -e .[dev]
test:
	PYTHONPATH=. pytest -q
lint:
	ruff check --select F,E9 --line-length 120 fullwave2_ultra tests examples
fixtures:
	PYTHONPATH=. python examples/tiny_nsims4/make_fixture.py
leak-check:
	bash scripts/check_no_private.sh
clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
