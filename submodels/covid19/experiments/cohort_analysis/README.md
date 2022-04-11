## Cohort Analysis

How do NH admissions compare to HCW attendance? If we quarentined every new NH admission from the community, would it have an impact?


#### Make Scenarios:

```
python submodels/covid19/experiments/cohort_analysis/src/make_scenario.py 
```

#### Run Scnearios:

Replace `n` and `cups` with the desired number of runs and cpus.

```
python submodels/covid19/experiments/cohort_analysis/src/run_scenario.py --n=10 --cpus=2
```

#### Analyze:

```
python submodels/covid19/experiments/cohort_analysis/src/analyze_scenarios.py
```
