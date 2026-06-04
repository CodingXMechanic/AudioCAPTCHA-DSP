.PHONY: help install test lint clean run

help:
	@echo "Targets: install, test, lint, clean, run"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	python -m compileall src/audiocaptcha_dsp/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov

run:
	audiocaptcha --help
