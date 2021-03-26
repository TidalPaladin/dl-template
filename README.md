# DL-Template

My deep learning template repository. 


## Overview of Features

The core components of this template are as follows:
* Premade utilities provided by [combustion](https://github.com/TidalPaladin/combustion)
* Dockerfiles that enable easy containerization of a newly developed 
  application
* CI/CD/devops features (provided via a Makefile) like code formatting,
  test running, etc.

This template is designed with following 3rd party libraries in mind:
* PyTorch-Lightning, a high level API for model training
* Hydra, a library that enables YAML based configuration of hyperparameters


## Installation

To pull the Combustion submodule and initialize a virtual environment, run

```
make init
```

## Usage

A project template is provided in `project`. Modify the source code in this directory
as needed. Configuration files in `conf` configure runtime parameters / hyperparameters.
Run `make demo` to run the existing code. A trivial training run on random input data
and labels will be executed on CPU.

### Configuration

Hydra allows for hyperparameters and runtime properties to be easily configured
using YAML files. The modularity of using multiple YAML files has many advantages
over command line configuration. See the Hydra 
[documentation](https://github.com/facebookresearch/hydra) for more details. 

Some noteable features of Hydra to be aware of are:
* Composition of multiple YAML files when specifying a runtime configuration
* Ability to specify YAML values at the command line
* Ability to specify YAML values at the command line


## Development

Multiple make recipes are provided to aid in development:
* `quality` - Runs code quality tests
* `style` - Automatically formats code (using `black` and `autopep8`)
* `test` - Runs all tests
* `test-%` - Runs specific tests by pattern match (via `pytest -k` flag)
* `test-pdb-%` - Runs specific tests with debugging on failure (via `pytest --pdb` flag)

## To Do
* CircleCI CI/CD pipeline


## References
* [PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning)
* [Hydra](https://github.com/facebookresearch/hydra)
* [Combustion](https://github.com/TidalPaladin/combustion)
* [Albumentations](https://albumentations.ai/)
