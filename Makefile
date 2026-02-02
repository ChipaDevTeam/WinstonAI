.PHONY: help install dev-install test lint format clean train trade

help:
	@echo "WinstonAI - Makefile Commands"
	@echo "=============================="
	@echo "install          Install production dependencies"
	@echo "dev-install      Install development dependencies"
	@echo "test             Run tests"
	@echo "lint             Run linters (flake8, pylint)"
	@echo "format           Format code with black"
	@echo "clean            Clean up generated files"
	@echo "train            Start model training"
	@echo "trade            Start live trading bot"
	@echo "gpu-check        Check GPU availability"
	@echo "requirements     Update requirements.txt"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

dev-install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ --max-line-length=100 --exclude=__pycache__
	pylint src/*.py --max-line-length=100 || true

format:
	black src/ tests/ --line-length=100
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ .coverage htmlcov/ .tox/

train:
	cd src && python train_gpu_optimized.py

train-quick:
	cd src && python quick_start_gpu.py

trade:
	cd src && python ultra_live_trading_bot.py

gpu-check:
	python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

gpu-monitor:
	cd src && python gpu_monitor.py

gpu-benchmark:
	cd src && python gpu_benchmark.py

requirements:
	pip freeze > requirements.txt

setup:
	python setup.py install

build:
	python -m build

docs:
	@echo "Documentation files are in the docs/ directory"
	@echo "Main README: README.md"
	@echo "Installation: docs/INSTALLATION.md"
	@echo "Configuration: docs/CONFIGURATION.md"
	@echo "API: docs/API.md"
