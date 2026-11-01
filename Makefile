.PHONY: install test build deploy deploy-prod delete clean

install:
	pip install -r requirements-dev.txt

test:
	python -m pytest -v tests/

build:
	sam build

deploy: build
	sam deploy

deploy-prod: build
	sam deploy --config-env prod

delete:
	sam delete --stack-name lambda-serverless-app-dev

clean:
	rm -rf .aws-sam .pytest_cache **/__pycache__