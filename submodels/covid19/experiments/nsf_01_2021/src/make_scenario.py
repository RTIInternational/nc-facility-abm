from pathlib import Path
from submodels.covid19.model.parameters import CovidParameters

import yaml

base_parameters_file = "submodels/covid19/experiments/base/scenario_base/parameters.yml"
experiment_dir = Path("submodels/covid19/experiments/nsf_01_2021/")


def main():
    """Setup the scenarios for the NSF runs."""

    base_vaccine_effectivness = 0.24
    base_case_multiplier = 8

    def make_params():
        # ----- Setup the base experiment
        with Path(base_parameters_file).open(mode="r") as f:
            params = yaml.safe_load(f)
        # ----- Turn on non-covid-hospitalizations
        params["track_hospitalizations"] = True
        params["use_historical_case_counts"] = True
        params["num_agents"] = 5_000_000
        params["baseline_vaccine_effectiveness"] = base_vaccine_effectivness
        params["new_vaccine_effectiveness"] = base_vaccine_effectivness
        params["case_multiplier"] = base_case_multiplier
        return params

    # Different Options
    options = {
        "base": {},
        "case_multiplier_4": {"case_multiplier": 4},
        "case_multiplier_6": {"case_multiplier": 6},
        "increase_vaccine_effectivness": {"new_vaccine_effectiveness": 0.75},
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

    # Special Scenario: Increase Asymptomatic Cases by 50%
    scenario_dir = experiment_dir.joinpath("scenario_increase_asymptomatic_cases")
    scenario_dir.mkdir(parents=True, exist_ok=True)
    params = make_params()
    covid_params = CovidParameters()
    for reported in ["reported", "nonreported"]:
        for vaccinated in ["notvaccinated", "vaccinated"]:
            p = f"{reported}_{vaccinated}_severity"
            cp = covid_params.__dict__[p]
            for age in [0, 1, 2]:
                cp[age][0] *= 1.5  # <--- This is the line that does +50%.
            params[p] = cp
    with scenario_dir.joinpath("parameters.yml").open(mode="w") as f:
        yaml.dump(params, f)


if __name__ == "__main__":
    main()
