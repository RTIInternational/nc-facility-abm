from submodels.covid19.model.covid import COVIDModel
from submodels.covid19.model.tests.conftest import test_dir, tests_setup, tests_teardown

if __name__ == "__main__":
    tests_setup(test_dir, num_agents=300000, time_horizon=30)
    self = COVIDModel(str(test_dir), run_dir="run_0", seed=1111)
    self.run_model()
    tests_teardown(test_dir)
