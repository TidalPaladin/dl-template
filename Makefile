.PHONY: clean clean-env check quality style tag-version test env upload upload-test

PROJECT=project
PY_VER=python3.10
PY_VER_SHORT=py$(shell echo $(PY_VER) | sed 's/[^0-9]*//g')
QUALITY_DIRS=$(PROJECT) tests setup.py
CLEAN_DIRS=$(PROJECT) tests
VENV=$(shell pwd)/env
PYTHON=$(VENV)/bin/python

LINE_LEN=120
DOC_LEN=120

CONFIG_FILE := config.mk
ifneq ($(wildcard $(CONFIG_FILE)),)
include $(CONFIG_FILE)
endif

check: ## run quality checks and unit tests
	$(MAKE) style
	$(MAKE) quality
	$(MAKE) types
	$(MAKE) test

clean: ## remove cache files
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*.pyc' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	find $(CLEAN_DIRS) -name '*.orig' -type f -delete

clean-env: ## remove the virtual environment directory
	rm -rf $(VENV)

init: ## pulls submodules and initializes virtual environment
	git submodule update --init --recursive
	$(MAKE) $(VENV)/bin/activate

node_modules: 
ifeq (, $(shell which npm))
	$(error "No npm in $(PATH), please install it to run pyright type checking")
else
	npm install
endif

quality: $(VENV)/bin/activate-quality
	$(MAKE) clean
	$(PYTHON) -m black --check --line-length $(LINE_LEN) --target-version $(PY_VER_SHORT) $(QUALITY_DIRS)
	$(PYTHON) -m flake8 --max-doc-length $(DOC_LEN) --max-line-length $(LINE_LEN) $(QUALITY_DIRS) 

style: $(VENV)/bin/activate-quality
	$(PYTHON) -m autoflake -r -i --remove-all-unused-imports --remove-unused-variables $(QUALITY_DIRS)
	$(PYTHON) -m isort $(QUALITY_DIRS)
	$(PYTHON) -m autopep8 -a -r -i --max-line-length=$(LINE_LEN) $(QUALITY_DIRS)
	$(PYTHON) -m black --line-length $(LINE_LEN) --target-version $(PY_VER_SHORT) $(QUALITY_DIRS)

test: $(VENV)/bin/activate-test ## run unit tests
	$(PYTHON) -m pytest \
		-rs \
		--cov=./$(PROJECT) \
		--cov-report=xml \
		./tests/

test-%: $(VENV)/bin/activate-test ## run unit tests matching a pattern
	$(PYTHON) -m pytest -rs -k $* -v ./tests/ 

test-pdb-%: $(VENV)/bin/activate-test ## run unit tests matching a pattern with PDB fallback
	$(PYTHON) -m pytest -rs --pdb -k $* -v ./tests/ 

test-ci: $(VENV)/bin/activate $(VENV)/bin/activate-test ## runs CI-only tests
	$(PYTHON) -m pytest \
		--cov=./$(PROJECT) \
		--cov-report=xml \
		-s -v \
		-m "not ci_skip" \
		./tests/

torch-fix:
	$(PYTHON) -m pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

types: $(VENV)/bin/activate node_modules
	npx --no-install pyright tests $(PROJECT)

upload: package
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository pypi dist/*

upload-test: package
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

env: $(VENV)/bin/activate ## create a virtual environment for the project

$(VENV)/bin/activate: setup.py setup.cfg requirements.txt
	test -d $(VENV) || $(PY_VER) -m venv $(VENV)
	$(PYTHON) -m ensurepip
	$(PYTHON) -m pip install -U pip 
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	touch $(VENV)/bin/activate

$(VENV)/bin/activate-%: $(VENV)/bin/activate
	$(PYTHON) -m pip install -e .[$*]
	touch $(VENV)/bin/activate-$*

help: ## display this help message
	@echo "Please use \`make <target>' where <target> is one of"
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}'
