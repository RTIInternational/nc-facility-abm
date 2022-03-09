# NSF Scenarios

#### Make Scenarios:

```
python submodels/covid19/experiments/nsf_01_2021/src/make_scenario.py 
```

#### Run Scnearios:

Replace `n` and `cups` with the desired number of runs and cpus.

```
python submodels/covid19/experiments/nsf_01_2021/src/run_scenario.py --n=1 --cpus=1
```

#### Analyze:

```
python submodels/covid19/experiments/nsf_01_2021/src/analyze_scenarios.py
```

## Server Runs

We ran this analysis on a server to make use of larger computer resources. 

* On the server:

```
docker-compose build
```

* Make & Run the scenarios:

```
docker-compose run -d hospital_abm bash -c "python3 submodels/covid19/experiments/nsf_01_2021/src/make_scenario.py"

docker-compose run -d hospital_abm bash -c "python3 submodels/covid19/experiments/nsf_01_2021/src/run_scenario.py --n=100 --cpus=15"

```

* It is possible that some runs fail. This seems to be a server issue, and not an error with the models themselves. The following code looks for directories that do not have results and reruns them with the same originally assigned seed. They do not fail a second time. 

```
docker-compose run -d hospital_abm bash -c "python3 submodels/covid19/experiments/nsf_01_2021/src/check_failures.py"
```

* Generate the aggregated output:

```
docker-compose run -d hospital_abm bash -c "python3 submodels/covid19/experiments/nsf_01_2021/src/analyze_scenarios.py"
```