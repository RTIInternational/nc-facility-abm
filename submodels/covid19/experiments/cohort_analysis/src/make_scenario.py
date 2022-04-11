import yaml
from pathlib import Path


base_parameters_file = "submodels/covid19/experiments/base/scenario_base/parameters.yml"
experiment_dir = Path("submodels/covid19/experiments/cohort_analysis")


if __name__ == "__main__":

    base_vaccine_effectivness = 0.24
    base_case_multiplier = 4

    def make_params():
        # ----- Setup the base experiment
        with Path(base_parameters_file).open(mode="r") as f:
            params = yaml.safe_load(f)
        # ----- Turn on non-covid-hospitalizations
        params["use_historical_case_counts"] = True
        params["num_agents"] = 2_500_000
        params["baseline_vaccine_effectiveness"] = base_vaccine_effectivness
        params["vaccine_effectiveness"] = base_vaccine_effectivness
        params["hcw_vaccine_effectiveness"] = base_vaccine_effectivness
        params["case_multiplier"] = base_case_multiplier
        params["track_hospitalizations"] = True
        return params

    # Different Options
    options = {
        "base": {},
    }
    for key, values in options.items():
        name = f"scenario_{key}"
        scenario_dir = experiment_dir.joinpath(name)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        params = make_params()
        for param_key, param_value in values.items():
            params[param_key] = param_value
        with scenario_dir.joinpath("parameters.yml").open(mode="w") as f:
            yaml.dump(params, f)
