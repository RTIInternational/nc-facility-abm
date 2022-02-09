import yaml
import shutil
from pathlib import Path


base_parameters_file = "submodels/covid19/experiments/base/scenario_base/parameters.yml"
experiment_dir = Path("submodels/covid19/experiments/nh_interventions")


def create_scenarios(name: str, parameter_name: str, values: list):

    for value in values:
        scenario_dir = experiment_dir.joinpath(f"scenario_{name}_{value}")
        scenario_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(base_parameters_file, scenario_dir)

        with scenario_dir.joinpath("parameters.yml").open(mode="r") as f:
            params = yaml.safe_load(f)
        params[parameter_name] = value
        with scenario_dir.joinpath("parameters.yml").open(mode="w") as f:
            yaml.dump(params, f)


if __name__ == "__main__":
    # ----- 1: Vary community vaccination
    create_scenarios(name="community_vaccines", parameter_name="community_vaccination", values=[0.65, 0.75, 0.85])

    # ----- 2: Healthcare worker vaccination
    create_scenarios(name="hcw_vaccination", parameter_name="healthcare_worker_vaccination", values=[0.70, 0.80, 0.90])

    # ----- 3: Lower Vaccine Effectiveness
    create_scenarios(name="vaccine_effectiveness", parameter_name="vaccine_effectiveness", values=[0.50, 0.60])

    # ----- 4: More Transmissable Variant
    create_scenarios(name="higher_tramission", parameter_name="r_effective", values=[1, 1.25, 1.5, 2])
