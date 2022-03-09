from submodels.covid19.model.covid import COVIDModel
from src.run_functions import run_failures
import argparse
from pathlib import Path


if __name__ == "__main__":
    description = """ Run a generic model. """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--cpus", type=int, default=1, help="number of CPUs to use simultaneously (default: %(default)s)"
    )
    args = parser.parse_args()
    cpus = args.cpus

    experiment_dir = Path("submodels/covid19/experiments/nh_interventions/")
    run_failures(COVIDModel, experiment_dir, cpus)
