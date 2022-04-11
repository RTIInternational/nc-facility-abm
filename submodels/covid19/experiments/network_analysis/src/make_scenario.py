from pathlib import Path

import yaml

import src.data_input as di

base_parameters_file = "submodels/covid19/experiments/base/scenario_base/parameters.yml"
experiment_dir = Path("submodels/covid19/experiments/network_analysis")

if __name__ == "__main__":

    base_vaccine_effectivness = 0.24
    base_case_multiplier = 6
    nc_populaiton = di.nc_counties().Population.sum()

    def make_params():
        # ----- Setup the base experiment
        with Path(base_parameters_file).open(mode="r") as f:
            params = yaml.safe_load(f)
        # ----- Turn on non-covid-hospitalizations
        params["use_historical_case_counts"] = True
        params["baseline_vaccine_effectiveness"] = base_vaccine_effectivness
        params["vaccine_effectiveness"] = base_vaccine_effectivness
        params["hcw_vaccine_effectiveness"] = base_vaccine_effectivness
        params["case_multiplier"] = base_case_multiplier
        return params

    # Different Options
    options = {
        "base_small": {"num_agents": 1_000_000},
        "contract_workers_2": {"contract_worker_n_sites": 2},
        "contract_workers_3": {"contract_worker_n_sites": 3},
        "contract_workers_4": {"contract_worker_n_sites": 4},
        "contract_workers_5": {"contract_worker_n_sites": 5},
        "increase_contract_hours_1.5": {"contract_hours_multiplier": 1.5},
        "decrease_contract_hours_.5": {"contract_hours_multiplier": 0.5},
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
