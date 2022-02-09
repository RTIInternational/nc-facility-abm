from collections import defaultdict
from typing import Union

import numpy as np
from src.jit_functions import first_true_value, sum_true

from model.state import LocationCategories


class BaseFacility:
    def __init__(self, beds: int):
        self.agents = set()
        self.model_int: int = None

        self.real_beds = {"total_beds": beds}
        self.model_beds = {"total_beds": beds}
        self._prep_facility(self.real_beds)

        self.single_site_full_time_workers = []
        self.single_site_part_time_workers = []
        self.multi_site_primary_workers = []
        self.multi_site_secondary_workers = []
        self.contract_workers = []
        self.worker_attendance = defaultdict(dict)

    def add_agent(self, unique_id: int) -> bool:
        """Adds an agent to an empty bed if available
        Args:
            unique_id (int): Agent ID
        Returns:
            bool: True if agent was added, False if no open beds
        """
        open_bed, bed_id = first_true_value(self.empty_beds())
        if open_bed:
            self.agents[bed_id] = unique_id
        return open_bed

    def remove_agent(self, unique_id: int) -> None:
        """Removes an agent from the facility by setting the value in the agents array to -1
        Args:
            unique_id (int): Unique Agent ID
        """
        self.agents[self.agents == unique_id] = -1

    def empty_beds(self) -> np.array:
        """Find empty beds meeting certain criteria. There is no criteria for the base facility class"""
        return self.open_beds()

    def open_beds(self) -> np.array:
        return self.agents < 0

    def apply_bed_multiplier(self, multiplier: float):
        self.model_beds = {
            "total_beds": max(1, round_to_int(self.real_beds["total_beds"] * multiplier)),
        }
        self._prep_facility(self.model_beds)

    def _prep_facility(self, beds: dict):
        """Set up the agents and boolean bed arrays."""
        self.agents = np.full(shape=beds["total_beds"], fill_value=-1, dtype=np.int64)

    def calculate_capacity(self):
        pass


class Community:
    def __init__(self):
        self.name = LocationCategories.COMMUNITY.name
        self.category = LocationCategories.COMMUNITY.name
        self.model_int = 0


class NursingHome(BaseFacility):
    def __init__(
        self, name: str, federal_provider_number: str, beds: int, avg_capacity: float, model_int: int, county_code: int
    ):
        super().__init__(beds)
        self.name = name
        self.federal_provider_number = federal_provider_number
        self.county_code = county_code
        self.category = LocationCategories.NH.name
        self.model_int = model_int
        self.avg_capacity = avg_capacity

    def calculate_capacity(self, count: bool = False) -> Union[float, int]:
        """Calculates percent full (capacity)
        Args:
            count (bool, optional): Return Count, rather than percent. Defaults to False.
        Returns:
            Union[float, int]: Capacity (percent or count occupied)
        """
        num = sum_true(~self.open_beds())
        den = self.model_beds["total_beds"]

        if count:
            return num
        else:
            if den == 0:
                return 0
            return num / den


class LongTermCareFacility(BaseFacility):
    def __init__(self, beds: int, name: str, model_int: int):
        super().__init__(beds)
        self.beds = beds
        self.name = name
        self.category = LocationCategories.LT.name
        self.model_int = model_int

    def calculate_capacity(self, count: bool = False) -> Union[float, int]:
        """Calculates percent full (capacity)
        Args:
            count (bool, optional): Return Count, rather than percent. Defaults to False.
        Returns:
            Union[float, int]: Capacity (percent or count occupied)
        """
        num = sum_true(~self.open_beds())
        den = self.model_beds["total_beds"]

        if count:
            return num
        else:
            if den == 0:
                return 0
            return num / den


class Hospital:
    def __init__(self, name: str, ncdhsr_name: str, county: str, n_beds: int, n_icu_beds: int):
        self.name = name
        self.ncdhsr_name = ncdhsr_name
        self.county = county
        self.category = LocationCategories.HOSPITAL.name
        self.real_beds = {"total_beds": n_beds, "icu_beds": n_icu_beds}
        self.real_beds["acute_beds"] = self.real_beds["total_beds"] - self.real_beds["icu_beds"]
        self.model_beds = {"total_beds": n_beds, "icu_beds": n_icu_beds}

        self._prep_facility(self.real_beds)

    def _prep_facility(self, beds: dict):
        """Set up the agents and boolean bed arrays."""
        self.agents = np.full(shape=beds["total_beds"], fill_value=-1, dtype=np.int64)
        self.icu_beds = _create_partial_boolean_array(beds["total_beds"], beds["icu_beds"])
        self.model_beds["acute_beds"] = self.model_beds["total_beds"] - self.model_beds["icu_beds"]

    def apply_bed_multiplier(self, multiplier: float):
        self.model_beds = {
            "total_beds": max(1, round_to_int(self.real_beds["total_beds"] * multiplier)),
            "icu_beds": round_to_int(self.real_beds["icu_beds"] * multiplier + 0.1),
        }

        self._prep_facility(self.model_beds)

    def empty_beds(self, icu: bool = False) -> np.array:
        """Find empty beds meeting certain criteria
        Args:
            icu (bool, optional): Beds need to be ICU. Defaults to False.
        Returns:
            np.array: array of `bool` values indicating bed availability by index
        """

        if icu:
            return self.open_beds() & self.icu_beds
        return self.open_beds() & ~self.icu_beds

    def open_beds(self) -> np.array:
        return self.agents < 0

    def add_agent(self, unique_id: int, icu: bool = False) -> bool:
        """Adds an agent to an empty bed in this Hospital (if available)
        Args:
            unique_id (int): Agent ID
            icu (bool, optional): Agent needs an ICU Bed. Defaults to False.
        Returns:
            bool: True if agent was added, False if no open beds
        """
        open_bed, bed_id = first_true_value(self.empty_beds(icu=icu))
        if open_bed:
            self.agents[bed_id] = unique_id
        return open_bed

    def remove_agent(self, unique_id: int) -> None:
        """Removes an agent from the hospital by setting the value in the agents array to -1
        Args:
            unique_id (int): Unique Agent ID
        """
        self.agents[self.agents == unique_id] = -1

    def calculate_capacity(self, capacity_type: str = "all", count: bool = False) -> Union[float, int]:
        """Calculates percent full (capacity)
        Args:
            capacity_type (str): One of ["all", "icu", "normal"]
            count (bool, optional): Return Count, rather than percent. Defaults to False.
        Returns:
            Union[float, int]: Capacity (percent or count occupied)
        """
        if capacity_type == "all":
            num = sum_true(~self.open_beds())
            den = self.model_beds["total_beds"]
        elif capacity_type == "icu":
            num = sum_true(self.icu_beds & ~self.open_beds())
            den = sum_true(self.icu_beds)
        elif capacity_type == "normal":
            num = sum_true(~self.icu_beds & ~self.open_beds())
            den = sum_true(~self.icu_beds)
        else:
            raise ValueError(f"capacity_type: {capacity_type} was provided. This type is not allowed.")

        if count:
            return num
        else:
            if den == 0:
                return 0
            return num / den


def _create_partial_boolean_array(length: int, n_filled: int):
    array = np.full(shape=length, fill_value=False)
    array[:n_filled] = True
    return array


def round_to_int(x: float) -> int:
    return int(round(x))
