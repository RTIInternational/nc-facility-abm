from pathlib import Path

import pytest
from model.tests.conftest import tests_setup, tests_teardown
from submodels.covid19.model.covid import COVIDModel

original_dir = Path("submodels/covid19/experiments/base/scenario_base")
test_dir = Path("submodels/covid19/experiments/base/scenario_test")


@pytest.fixture(scope="class")
def model(request):
    """Use this fixture if you want a fresh copy of a test model"""
    tests_setup(test_dir, 500_000, time_horizon=30)
    request.cls.model = COVIDModel(scenario_dir=str(test_dir), run_dir="run_0")
    tests_teardown(test_dir)


@pytest.fixture(scope="class")
def model_small(request):
    """Use this fixture if you want a fresh copy of a test model"""
    tests_setup(test_dir, 100_000, time_horizon=30)
    request.cls.model = COVIDModel(scenario_dir=str(test_dir), run_dir="run_0")
    tests_teardown(test_dir)


@pytest.fixture(scope="class")
def model_with_run(request):
    """Use this fixture if you want to run the model in addition to initializing it"""
    tests_setup(test_dir, 1_500_000, time_horizon=30)
    model = COVIDModel(scenario_dir=str(test_dir), run_dir="run_0")
    model.run_model()
    request.cls.model = model
    tests_teardown(test_dir)


@pytest.fixture(scope="class")
def xl_model_with_run(request):
    """
    Same as model_with_run but with a larger population (for tests which may be adversely
    affected by an unrealistically small multiplier) and shorter horizon (to reduce compute)
    """
    tests_setup(test_dir, 10_000_000, time_horizon=10)
    model = COVIDModel(scenario_dir=str(test_dir), run_dir="run_0")
    model.run_model()
    request.cls.model = model
    tests_teardown(test_dir)
