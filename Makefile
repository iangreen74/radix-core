.PHONY: install install-dev test lint format typecheck security clean docker-build help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install with all dev dependencies
	pip install -e ".[all]"
	pip install pytest pytest-cov pytest-mock pytest-asyncio hypothesis ruff black isort mypy bandit

test: ## Run tests
	pytest tests/ -x -q --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=radix_core --cov-report=term-missing --cov-fail-under=70

lint: ## Run linter
	ruff check radix_core/ tests/

format: ## Check code formatting
	black --check radix_core/ tests/
	isort --check-only radix_core/ tests/

format-fix: ## Auto-fix formatting
	black radix_core/ tests/
	isort radix_core/ tests/

typecheck: ## Run type checker
	mypy radix_core/ --ignore-missing-imports

security: ## Run security scanner
	bandit -r radix_core/ -c pyproject.toml 2>/dev/null || bandit -r radix_core/ -s B101

docker-build: ## Build Docker images
	docker build -t radix-core:dev .
	docker build -t radix-dashboard:dev agents/dashboard/

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage .mypy_cache/
