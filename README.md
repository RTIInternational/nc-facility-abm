# Agent-Based Model Framework for the North Carolina Modeling Infectious Diseases Program (NC MInD ABM) 

This repository contains a geospatial hospitlization ABM for North Carolina. Using public data only, we have created a model that simulates hospitalizations and nursing home stays for agents using [RTI SynthPop™](https://www.rti.org/synthpop-synthetic-population-data-analysis).

### Environment Setup

If you already have python3.8 and pip installed on your computer, you can run the following from the repo directory:

```bash
pip install virtualenv
virtualenv python_env --python=python3.8
source python_env/bin/activate
pip install -r docker/requirements.txt
pip install -e .
pip install docker/optabm
```

You can also setup a docker container for this project.

```
docker-compose build
```

### 🔧 Tests

Note that this will only test the base hospital model. Submodels should be tested on their own.

To run all unit tests:

```bash
pytest model/ --cov=model --ignore=docker --verbose
```

For an interactive html report, add `--cov-report html` to the end.


#### 📊 Profiling via `pyinstrument`

In addition to running unit tests, we use profiling to identify slow functions in the code. The following will produce a report on model execution time:

```bash
sudo pyinstrument --html -o profiletest.html src/run_examples.py 2
```


### 🧬 Adding a Disease Model 

The hospital abm on its own will only recreate agent movement through the various location nodes in the model. To use this in practice, a disease model should be added. Please see this [README.md](submodels/covid19/README.md) for an example.


### Different types of experiment/scenario runs

There are several different methods for running ABM scenarios. You may want 100 runs of the same parameters but using different seeds, you may want 100 runs varrying parameters. We provide two quick examples of this. 

**Example #0**: Run the same scenario multiple times with different seeds.

```
python src/run_examples.py 0
```

**Example #1**: Run all scenarios within an experiment folder.

```bash
python src/run_examples.py 1
```