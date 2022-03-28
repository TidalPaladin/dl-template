.PHONY: docker clean clean-venv check ci-test pre-commit quality run style tag-version test venv upload upload-test

PROJECT=project
PY_VER=python3.10
PY_VER_SHORT=py$(shell echo $(PY_VER) | sed 's/[^0-9]*//g')
QUALITY_DIRS=$(PROJECT) tests setup.py
CLEAN_DIRS=$(PROJECT) tests
VENV=$(shell pwd)/env
PYTHON=$(VENV)/bin/python

LINE_LEN=120
DOC_LEN=120

VERSION := $(shell cat version.txt)

CONFIG_FILE := config.mk
ifneq ($(wildcard $(CONFIG_FILE)),)
include $(CONFIG_FILE)
endif


check: ## runs quality/style checks and tests
	$(MAKE) style
	$(MAKE) quality
	$(MAKE) types
	$(MAKE) test

ci-test: $(VENV)/bin/activate-test
	$(PYTHON) -m pytest \
		-rs \
		--cov=./src \
		--cov-report=xml \
		-s -v \
		-m "not ci_skip" \
		./tests/

clean: 
	find $(CLEAN_DIRS) -path '*/__pycache__/*' -delete
	find $(CLEAN_DIRS) -type d -name '__pycache__' -empty -delete
	find $(CLEAN_DIRS) -name '*@neomake*' -type f -delete
	find $(CLEAN_DIRS) -name '*.pyc' -type f -delete
	find $(CLEAN_DIRS) -name '*,cover' -type f -delete
	find $(CLEAN_DIRS) -name '*.orig' -type f -delete

clean-venv:
	rm -rf $(VENV)


init:
	git submodule update --init --recursive
	$(MAKE) venv

node_modules:
	npm install

quality: $(VENV)/bin/requirements.quality.txt
	$(MAKE) clean
	$(PYTHON) -m black --check --line-length $(LINE_LEN) --target-version $(PY_VER_SHORT) --exclude $(PROJECT)/version.py $(QUALITY_DIRS)
	$(PYTHON) -m flake8 --max-doc-length $(DOC_LEN) --max-line-length $(LINE_LEN) $(QUALITY_DIRS) 

reset:
	$(MAKE) clean
	$(MAKE) clean-venv
	$(MAKE) check

fit: $(CONFIG_FILE)
	$(PYTHON) -m $(PROJECT) fit --config config.yaml

style: $(VENV)/bin/requirements.quality.txt
	$(PYTHON) -m autoflake -r -i --remove-all-unused-imports --remove-unused-variables $(QUALITY_DIRS)
	$(PYTHON) -m isort $(QUALITY_DIRS)
	$(PYTHON) -m autopep8 -a -r -i --max-line-length=$(LINE_LEN) $(QUALITY_DIRS)
	$(PYTHON) -m black --line-length $(LINE_LEN) --target-version $(PY_VER_SHORT) $(QUALITY_DIRS)


test: $(VENV)/bin/requirements.dev.txt
	$(PYTHON) -m pytest \
		-rs \
		--cov=./$(PROJECT) \
		--cov-report=xml \
		--cov-report=term \
		./tests/

test-%: $(VENV)/bin/requirements.dev.txt
	$(PYTHON) -m pytest -rs -k $* -s -v ./tests/ 

test-ci: $(VENV)/bin/requirements.dev.txt ## runs CI-only tests
	$(PYTHON) -m pytest \
		--cov=./$(PROJECT) \
		--cov-report=xml \
		--cov-report=term \
		-s -v \
		-m "not ci_skip" \
		./tests/

test-pdb-%: $(VENV)/bin/requirements.dev.txt
	$(PYTHON) -m pytest -rs --pdb -k $* -s -v ./tests/ 

torch-fix: ## installs torch with cuda11.3 support (for Ampere GPUs)
	$(PYTHON) -m pip install \
		torch==1.11.0+cu113 \
		torchvision==0.12.0+cu113 \
		torchaudio==0.11.0+cu113 \
		-f https://download.pytorch.org/whl/cu113/torch_stable.html

types: $(VENV)/bin/requirements.dev.txt node_modules ## checks types with pyright
	npx --no-install pyright -p pyrightconfig.json

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: setup.py requirements.txt 
	test -d $(VENV) || python3 -m virtualenv -p $(PY_VER) $(VENV)
	$(PYTHON) -m pip install -U pip 
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	touch $(VENV)/bin/activate

$(VENV)/bin/requirements.%.txt: requirements.%.txt
	test -d $(VENV) || python3 -m virtualenv -p $(PY_VER) $(VENV)
	$(PYTHON) -m pip install -U pip 
	$(PYTHON) -m pip install -r $<
	touch $(VENV)/bin/$<
