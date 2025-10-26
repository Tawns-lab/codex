# Makefile for Adversarial Forensic Testing

.PHONY: help install test demo full-test clean fixtures lint format

help:
	@echo "Adversarial Forensic Testing - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "  make install        - Install dependencies"
	@echo "  make fixtures       - Generate test fixtures"
	@echo "  make demo           - Run simple demo"
	@echo "  make test           - Run unit tests"
	@echo "  make full-test      - Run full adversarial test suite"
	@echo "  make advanced-test  - Run advanced adversarial tests (NEW!)"
	@echo "  make clean          - Clean generated files"
	@echo "  make lint           - Run code linters"
	@echo "  make format         - Format code"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

fixtures:
	@echo "Generating test fixtures..."
	cd src && python data_generators.py

demo:
	@echo "Running simple demo..."
	cd examples && python simple_demo.py

test:
	@echo "Running unit tests..."
	cd tests && python -m pytest test_forensic_engine.py -v

full-test:
	@echo "Running full adversarial test suite..."
	@echo "This may take several minutes..."
	cd examples && python run_full_test_suite.py --output-dir ../test_results

advanced-test:
	@echo "Running advanced adversarial test suite..."
	@echo "This includes sophisticated attack strategies..."
	cd examples && python run_advanced_adversarial_tests.py --output-dir ../test_results/advanced

clean:
	@echo "Cleaning generated files..."
	rm -rf test_results/
	rm -rf data/*.pt data/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete"

lint:
	@echo "Running linters..."
	@command -v pylint >/dev/null 2>&1 && pylint src/ || echo "pylint not installed"
	@command -v flake8 >/dev/null 2>&1 && flake8 src/ --max-line-length=100 || echo "flake8 not installed"

format:
	@echo "Formatting code..."
	@command -v black >/dev/null 2>&1 && black src/ tests/ examples/ || echo "black not installed"

# Development commands
dev-setup: install fixtures
	@echo "Development environment ready!"

all: install fixtures test demo
	@echo "All setup and tests complete!"
