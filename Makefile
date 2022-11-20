test:
	python -m pytest --cov=peg --cov-report=html --cov-report=term tests

clean:
	rm -rf dist

build: clean
	python -m build --wheel

publish: build
	twine upload dist/*
