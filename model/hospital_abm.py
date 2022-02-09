import random
from pathlib import Path

import numpy as np
import src.data_input as di
import tqdm
from src.data_input import sample_population
from src.jit_functions import assign_conditions, init_daily_state, insert_daily_state
from src.misc_functions import get_multiplier

from model.life import Life
from model.movement import Movement
from model.north_carolina_nodes import NcNodeCollection
from model.parameters import ParameterContainer, Parameters
from model.prepare_transitions import prepare_transitions


class HospitalABM:
    def __init__(self, scenario_dir: str, run_dir: str, seed: int = 1111, params: ParameterContainer = Parameters):
        """An Agent-based model of North Carolina Hospitals"""
        # ----- Setup Randominess
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rng_stdlib = random.Random(self.seed)

        # ----- Load parameters
        self.scenario_dir = Path(scenario_dir)
        self.run_dir = self.scenario_dir.joinpath(run_dir)
        self.run_dir.mkdir(exist_ok=True)
        self.output_dir = self.run_dir.joinpath("model_output")
        self.output_dir.mkdir(exist_ok=True)

        self.params = params()
        self.params.update_from_file(self.scenario_dir.joinpath("parameters.yml"))
        self.seed = seed

        # ----- Create the Population
        self.population = sample_population(
            variables=tuple(self.params.synpop_variables),
            limit=self.params.num_agents,
            seed=self.seed,
        )
        self.unique_ids = np.arange(self.params.num_agents, dtype=np.int32)

        for column in self.population.columns:
            setattr(self, column.lower(), self.population[column].values)

        self.county_population = di.nc_counties().set_index("County")["Population"].to_dict()
        self.county_codes_dict = di.nc_counties().set_index("County_Code")["County"].to_dict()
        self._county_code_mask = {code: self.county_code == code for code in self.params.counties}
        self._county_code_unique_ids = {c: self.unique_ids[self._county_code_mask[c]] for c in self.params.counties}

        # Assign concurrent conditions
        setattr(self, "concurrent_conditions", assign_conditions(self.age_group, self.rng.rand(len(self.population))))

        # Prepare transitions based on parameters
        for key, value in prepare_transitions(self.params).items():
            setattr(self, key, value)

        # ----- Load the nodes
        self.time = 0
        self.multiplier = get_multiplier(self.params)
        self.nodes = NcNodeCollection(multiplier=self.multiplier)

        # ----- Load the movement module
        self.movement = Movement(self)

        # ----- Load the life module
        self.life = Life(self)

        # ----- Data Collection
        self.daily_state_data = init_daily_state(2, self.nodes.number_of_items, self.params.time_horizon + 1)

        # ----- List of Update for Agents
        self.actions = []
        self.insert_daily()

    def run_model(self, print_status=True):
        step_range = range(0, self.params.time_horizon)
        if print_status:
            step_range = tqdm.trange(0, self.params.time_horizon, desc="---> Model running")
        for _ in step_range:
            self.step()

    def step(self):
        self.time += 1
        self.life.step()
        self.movement.step()
        self.perform_actions()
        self.insert_daily()

    def insert_daily(self):
        if self.params.collect_daily_data:
            insert_daily_state(self.life.values, self.movement.location.values, self.time, self.daily_state_data)

    def perform_actions(self):
        self.rng.shuffle(self.actions)
        for action in self.actions:
            self.current_action = action
            action[0](**action[1])
        self.actions = []

    def probabilistic_round(self, x):
        return int(x + self.rng.rand())
