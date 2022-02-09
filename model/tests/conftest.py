import shutil
from pathlib import Path

import pytest
import yaml
from model.hospital_abm import HospitalABM


original_dir = Path("experiments/base/scenario_base")
test_dir = Path("experiments/base/scenario_test")


def tests_setup(test_dir: Path = test_dir, num_agents: int = 1_000_000, time_horizon: int = 100):

    # ----- Remove existing directory
    tests_teardown(test_dir)

    # ----- Copy Parameters to Test Dir
    test_dir.mkdir(parents=True)
    shutil.copy(original_dir.joinpath("parameters.yml"), test_dir)

    with test_dir.joinpath("parameters.yml").open(mode="r") as f:
        params = yaml.safe_load(f)

    params["num_agents"] = num_agents
    params["time_horizon"] = time_horizon

    with test_dir.joinpath("parameters.yml").open(mode="w") as f:
        yaml.dump(params, f)


def tests_teardown(test_dir):
    if test_dir.is_dir():
        shutil.rmtree(test_dir)


@pytest.fixture(scope="class")
def model(request):
    """Use this fixture if you want a fresh copy of a test model"""
    tests_setup(num_agents=500_000)
    request.cls.model = HospitalABM(scenario_dir=str(test_dir), run_dir="run_0")
    tests_teardown(test_dir)


@pytest.fixture(scope="class")
def model_small(request):
    """Use this fixture if you want a fresh copy of a test model"""
    tests_setup(num_agents=100_000)
    request.cls.model = HospitalABM(scenario_dir=str(test_dir), run_dir="run_0")
    tests_teardown(test_dir)


@pytest.fixture(scope="session")
def model_with_run(request):
    """Use this fixture if you want a fresh copy of an LDM model to execute a test.

    Use instead of copy(model) inside of each test"""
    tests_setup(num_agents=500_000)
    model = HospitalABM(scenario_dir=str(test_dir), run_dir="run_0")
    model.run_model()
    request.cls.model = model
    tests_teardown(test_dir)
