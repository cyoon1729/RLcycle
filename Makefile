format:
	black .
	isort -rc .
	env PYTHONPATH=. pytest --pylint --flake8
