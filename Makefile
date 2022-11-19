test:
	python -m pytest --cov=peg --cov-report=html --cov-report=term tests

build:
	python -m build --wheel
