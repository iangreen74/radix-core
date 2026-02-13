.PHONY: install test lint clean

install:
	pip install -e .

test:
	pytest tests/ -x -q

lint:
	ruff check radix_core/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage
