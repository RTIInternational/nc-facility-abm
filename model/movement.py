from copy import copy
from functools import lru_cache

import numpy as np
import pandas as pd
import src.data_input as di
from optabm.example_functions import init_distribution
from optabm.misc_functions import normalize, normalize_and_create_cdf
from optabm.optapply import check_agents_with_array
from optabm.state import State, StateEnum
from optabm.storage import EventStorage
from scipy.special import expit
from scipy.stats import truncnorm
from src.jit_functions import update_community_probability

from model.facilities import Hospital
from model.state import AgeGroup, LifeState, LocationCategories


class Movement:
    def __init__(self, model):
        self.model = model
        self.params = model.params

        # ----- Empty Lists/Dicts to help with movement
        self.current_los = dict()
        self.leave_facility_day = dict()
        self.last_movement_day = dict()
        self.readmission_date = dict()
        self.readmission_location = dict()

        self.facilities = model.nodes.facilities
        self.community = model.nodes.community

        # ----- An EventStorage container for patients turned away
        self.patients_turned_away = EventStorage(column_names=["Unique_ID", "Time", "Location", "ICU"])

        # ----- Probability of Community to Hospital Movement
        keys = list(zip(model.county_code, model.age_group))
        temp_map = {row.Index: row.HOSPITAL for row in self.model.community_transitions.itertuples()}
        self.community_to_hospital_probabilities = np.zeros(self.model.params.num_agents, dtype=np.float)
        for i in range(len(keys)):
            self.community_to_hospital_probabilities[i] = temp_map[keys[i]]
        # Update value based on concurrent conditions
        self.community_to_hospital_probabilities = update_community_probability(
            self.community_to_hospital_probabilities, self.model.age_group, self.model.concurrent_conditions
        )
        # ----- Probability of Community to NH Movement
        temp_map = {row.Index: row.NH for row in self.model.community_transitions.itertuples()}
        self.community_to_nh_probabilities = np.zeros(self.model.params.num_agents, dtype=np.float)
        for i in range(len(keys)):
            self.community_to_nh_probabilities[i] = temp_map[keys[i]]

        # ----- Probability of next facility type when leaving a facility
        self.facility_transitions = location_dict(self.model, self.model.facility_transitions)

        # ----- Probability of next hospital
        self.discharges = self.model.discharges_df.copy()
        # Switch column names to ints
        self.discharges.columns = [self.model.nodes.name_to_int[item] for item in self.discharges.columns]
        # Switch counts to probabilities
        self.discharges = self.discharges.div(self.discharges.sum(axis=1), axis=0).fillna(0)

        # ----- Hospitals by county: Create a list of hospitals that supply each county
        dis = self.discharges
        self.hospitals_by_county = {county: list(dis.columns[dis.loc[county] > 0]) for county in list(dis.index)}
        # Transfer probabilities, by facility, by county
        self.add_distributions_for_transfers()

        # ----- Distance to Facility Information
        self.county_to_hospital_distances = di.county_hospital_distances()
        self.county_nh_probabilities = di.county_facility_distribution(
            LocationCategories.NH.name,
            closest_n=model.params.location.NH_closest_n,
        )
        self.county_lt_probabilities = di.county_facility_distribution(
            LocationCategories.LT.name, closest_n=model.params.location.LT_closest_n
        )
        # ----- Length of Stay Distributions
        self.los = {
            "NH": di.nh_los(),
            "LT": self.init_los_from_gamma(los_info=self.params.location.LT_LOS),
        }
        self.initial_los_distributions = {
            "NH": init_distribution(self.model.rng, self.los["NH"]),
            "LT": init_distribution(self.model.rng, self.los["LT"]),
        }
        for hospital_id in self.model.nodes.category_ints["HOSPITAL"]:
            mean = self.model.hospital_df.loc[self.model.nodes.facilities[hospital_id].name].LOS
            self.los[hospital_id] = self.init_los_from_mean(mean)
            self.initial_los_distributions[hospital_id] = init_distribution(self.model.rng, self.los[hospital_id])

        # ----- Location State
        LocationState = StateEnum(
            "Location", [i.name.replace(" ", "_") for i in self.model.nodes.facilities.values()], start=0
        )
        self.location = State(self.model, state_enum=LocationState)
        self.location.values.fill(self.model.nodes.community)
        columns = ["Unique_ID", "Time", "Location", "New_Location", "ICU", "LOS"]
        self.location.state_changes = EventStorage(columns, store_events=self.model.params.location.store_events)
        # ----- Assign LOS based on starting location and update location values
        self.init_nh_ltach_agents()
        self.init_hospitals()
        self.location.previous = self.location.values.copy()

    def step(self):
        """Step through 1 day of location updates
        Collect a set of unique_ids and actions to perform. Randomize. Execute
        """
        self.facility_movement()
        self.community_movement()
        # self.readmission_movement()

    def init_nh_ltach_agents(self):
        """For agents starting in NH and LTs, determing LOS and assign agent to facility"""
        county_p = {}
        county_counts = {}
        for category in [LocationCategories.NH.name, LocationCategories.LT.name]:
            facility_to_county_p = di.facility_to_county_probabilities(category)
            for key in list(facility_to_county_p.keys()):
                facility_to_county_p[self.model.nodes.name_to_int[key]] = facility_to_county_p.pop(key)
                county_p[category] = facility_to_county_p
            county_counts[category] = {c: [] for c in self.model.params.counties}

        # Assign the number of NH and LT to get from each county for each location
        for category in [LocationCategories.NH.name, LocationCategories.LT.name]:
            for location in self.model.nodes.category_ints[category]:
                facility = self.facilities[location]
                fill_proportion = self.params.location.lt_fill_proportion
                if category == LocationCategories.NH.name:
                    fill_proportion = facility.avg_capacity / facility.real_beds["total_beds"]
                prob_list = county_p[category][location]
                beds = max(1, int(round((facility.model_beds["total_beds"] * fill_proportion))))
                selected_counties = self.model.rng_stdlib.choices(prob_list[1], cum_weights=prob_list[0], k=beds)
                for county in selected_counties:
                    county_counts[category][county].append(location)

        # Sample an actual person in the community from the county
        for category in county_counts.keys():
            for county in county_counts[category]:
                locations = county_counts[category][county]
                if len(locations) == 0:
                    continue
                unique_ids = self.model.unique_ids[(self.model.county_code == county) & (self.location.values == 0)]
                # NH is 65+ Only
                if category == LocationCategories.NH.name:
                    unique_ids = unique_ids[self.model.age_group[unique_ids] == 2]
                    if len(unique_ids) > 0:
                        size = min(len(locations), len(unique_ids))
                        unique_ids = self.model.rng.choice(unique_ids, size=size, replace=False)
                # LT should be 75% 65+ and 25% 50-65
                if category == LocationCategories.LT.name:
                    temp_ids2 = unique_ids[self.model.age_group[unique_ids] == 2]
                    temp_ids1 = unique_ids[self.model.age_group[unique_ids] == 1]
                    max_ids1 = int(len(temp_ids2) * 0.33)
                    if len(temp_ids1) > max_ids1:
                        temp_ids1 = temp_ids1[0:max_ids1]
                    unique_ids = np.concatenate((temp_ids1, temp_ids2))
                    if len(unique_ids) > 0:
                        size = min(len(locations), len(unique_ids))
                        unique_ids = self.model.rng.choice(unique_ids, size=size, replace=False)
                for _, unique_id in enumerate(unique_ids):
                    location = locations[_]
                    self.assign_los(unique_id=unique_id, new_location=location, initialize=True)
                    self.location.values[unique_id] = location
                    self.facilities[location].add_agent(unique_id)

    def init_los_from_gamma(self, los_info: dict):
        """Initialize length of stays options"""
        initial_sample = self.model.rng.gamma(los_info["shape"], los_info["support"], size=10000)
        return [int(i) for i in np.rint(initial_sample)]

    def init_los_from_mean(self, mean: float):
        """Create a truncated normal for the mean los of a location.
        Add 1 to go from "number of nights" to "number of days"
        """
        lower, upper = 1, mean * 5
        mu, sigma = mean, (mean / 2)
        np.random.seed(seed=0)
        distribution = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        values = distribution.rvs(10_000)
        return list(np.rint(values).astype(int))

    def init_hospitals(self):
        """For agents starting in a hospital, determine LOS and ICU assignment."""
        # A dictionary to hold the facilities needing someone from this county/age combination
        county_age_normal = {(i, j): [] for i in self.model.params.counties for j in self.model.params.age_groups}
        county_age_icu = {(i, j): [] for i in self.model.params.counties for j in self.model.params.age_groups}

        for _, facility in self.facilities.items():
            if facility.category == LocationCategories.HOSPITAL.name:
                self.assign_non_nc_placeholders(facility)
                self.select_facility(facility, county_age_dict=county_age_normal, icu=False)
                self.select_facility(facility, county_age_dict=county_age_icu, icu=True)

        # Find appropriate agents
        self._init_in_community = self.location.values == self.community
        for key, value in county_age_normal.items():
            self.assign_agents(key, value, icu=False)
        self._init_in_community = self.location.values == self.community
        for key, value in county_age_icu.items():
            self.assign_agents(key, value, icu=True)

    def assign_agents(self, key: int, values: list, icu: bool):
        # From County, of specific age group, & in community
        filter1 = self.model._county_code_mask[key[0]]
        filter2 = self.model.age_group == key[1]
        unique_ids = self.model.unique_ids[filter1 & filter2 & self._init_in_community].tolist()
        # Sample IDS
        if len(unique_ids) > 0:
            unique_ids = self.model.rng_stdlib.sample(list(unique_ids), k=len(values))
            for i, unique_id in enumerate(unique_ids):

                facility = self.facilities[values[i]]
                self.assign_los(unique_id=unique_id, new_location=facility.model_int, initialize=True)

                self.location[unique_id] = facility.model_int
                facility.add_agent(unique_id, icu=icu)

    def community_movement(self):
        """Randomly select agents in the community to move to a facility"""
        # Community to STACH
        living = self.model.life.values == LifeState.ALIVE
        community = self.location.values == self.community
        use_agents = self.model.unique_ids[living & community]
        probabilities = self.community_to_hospital_probabilities[use_agents]
        selected_ids_stach = check_agents_with_array(self.model, use_agents, probabilities)
        for unique_id in selected_ids_stach:
            self.model.actions.append([self.move_to_stach, {"unique_id": unique_id}])
        # Community to NH
        probabilities = self.community_to_nh_probabilities[use_agents]
        selected_ids = check_agents_with_array(self.model, use_agents, probabilities)
        for unique_id in selected_ids:
            if unique_id not in selected_ids_stach:
                self.model.actions.append([self.move_to_nh, {"unique_id": unique_id}])

    def facility_movement(self):
        """Move all agents not in the community whose LOS ends today"""
        unique_ids = [key for key, value in self.leave_facility_day.items() if value == self.model.time]
        for unique_id in unique_ids:
            self.model.actions.append([self.location_update, {"unique_id": unique_id}])

    def location_update(self, unique_id: int):
        """Update an agents location based on probability of next location"""
        current_location = self.location[unique_id]
        current_category = self.facilities[current_location].category
        # Agents have a chance of dying
        if self.model.rng.rand() < self.model.death_dictionary[current_category]:
            self.model.life.death(unique_id)
            return
        previous_category = self.facilities[self.location.previous[unique_id]].category
        self.u_id = unique_id
        age = self.model.age_group[unique_id]
        county = self.model.county_code[unique_id]

        # 80% of previous NH patients (who are leaving a hospital) must return to a NH
        if (previous_category == LocationCategories.NH.name) and (current_category == LocationCategories.HOSPITAL.name):
            if self.model.rng.rand() < self.params.location.nh_st_nh:
                new_category = LocationCategories.NH.name
            else:
                p = self.find_location_transitions(county, 1, current_location)  # Use age=1 to force a non NH movement
                new_category = self.model.rng_stdlib.choices(self.model.nodes.categories, cum_weights=p)[0]
        else:
            p = self.find_location_transitions(county, age, current_location)
            new_category = self.model.rng_stdlib.choices(self.model.nodes.categories, cum_weights=p)[0]

        if new_category == LocationCategories.COMMUNITY.name:
            if current_category == LocationCategories.HOSPITAL.name:
                if self.model.rng.rand() < self.params.location.readission:
                    day = self.model.rng.randint(2, self.params.location.readmission_days)
                    self.readmission_date[unique_id] = self.model.time + day
                    self.readmission_location[unique_id] = current_location
            self.go_home(unique_id, current_location)
        elif new_category in LocationCategories.HOSPITAL.name:
            self.move_to_stach(unique_id, current_location)
        elif new_category == LocationCategories.NH.name:
            self.move_to_nh(unique_id, current_location)
        elif new_category == LocationCategories.LT.name:
            self.move_to_lt(unique_id, current_location, county)

    def move_to_stach(self, unique_id: int, current_location: int = 0, force_icu: bool = False, test_icu: bool = True):
        """Move patient to a STACH. This function determines if an ICU or non-ICU bed will be used
        We assume the patient is coming from the community unless current_location is specified.
        """
        # First location
        county = self.model.county_code[unique_id]
        ids = self.facilities[current_location].hospital_options[county]["ids"]
        probs = self.facilities[current_location].hospital_options[county]["cdf"]
        new_location = self.model.rng_stdlib.choices(ids, cum_weights=probs, k=1)[0]

        # -----  Determine if this is an ICU patient
        potential_los = self.select_los(new_location)
        icu = False
        if test_icu & (self.facilities[new_location].model_beds["icu_beds"] > 0):
            age = self.model.age_group[unique_id]
            size = self.facilities[new_location].model_beds["total_beds"]
            cc = self.model.concurrent_conditions[unique_id]
            p = self.find_icu_probability(age, size, potential_los, cc) * self.params.location.icu_reduction_multiplier
            if self.model.rng.rand() < p:
                icu = True
        if force_icu:
            icu = True

        self.stach_loop(unique_id, current_location, new_location, county, icu, potential_los)

    def stach_loop(self, unique_id: int, current_location: int, new_location: int, county: int, icu: bool, los: int):
        """Try to move a patient to the selected new_location"""
        # If space, move them to the hospital (try both ICU and non-ICU beds)
        if self.facilities[new_location].add_agent(unique_id, icu):
            self.assign_los(unique_id, new_location, initialize=False, los=los)
            self.change_location(unique_id, current_location, new_location, icu=icu)
        elif self.facilities[new_location].add_agent(unique_id, icu=(not icu)):
            self.assign_los(unique_id, new_location, initialize=False, los=los)
            self.change_location(unique_id, current_location, new_location, icu=(not icu))
        # If not and patient is a transfer, send them home. Should not have tried a transfer if hospital was full.
        elif current_location != self.community:
            self.go_home(unique_id, current_location=current_location)
        # Person is turned away - Try to find an equal bed somewhere else
        else:
            # Try all hospitals the agent would normaly try
            ids = self.facilities[current_location].hospital_options[county]["ids"]
            probs = self.facilities[current_location].hospital_options[county]["prob_array"]
            new_locations = self.model.rng.choice(ids, size=len(ids), replace=False, p=probs)
            for new_location in new_locations:
                if self.facilities[new_location].add_agent(unique_id, icu=icu):
                    self.assign_los(unique_id, new_location, initialize=False, los=los)
                    self.change_location(unique_id, current_location, new_location, icu=icu)
                    return
            # If we still haven't moved, any other hospital by distance from county centroid
            items = self.county_to_hospital_distances[county]
            new_locations = [f for f in items if f["distance_mi"] <= self.params.location.max_distance]
            for new_location in new_locations:
                if new_location["Name"] in self.model.nodes.name_to_int:
                    new_location = self.model.nodes.name_to_int[new_location["Name"]]
                    if self.facilities[new_location].add_agent(unique_id, icu=icu):
                        self.assign_los(unique_id, new_location, initialize=False, los=los)
                        self.change_location(unique_id, current_location, new_location, icu=icu)
                        return
            self.patients_turned_away.record_event((unique_id, self.model.time, new_location, int(icu)))

    def readmission_movement(self):
        """Model readmission from the community. Only move patient if they are still at home.
        Some patients will return before their readmission date - they should not be moved.
        """
        unique_ids = [key for key, value in self.readmission_date.items() if value == self.model.time]
        for unique_id in unique_ids:
            self.model.actions.append([self.simulate_readmission, {"unique_id": unique_id}])

    def simulate_readmission(self, unique_id: int):
        current_location = self.location[unique_id]
        if current_location == 0:
            if self.model.life.values[unique_id] == LifeState.ALIVE:
                new_location = self.readmission_location[unique_id]
                county = self.model.county_code[unique_id]
                los = self.select_los(new_location)
                icu = False
                self.stach_loop(unique_id, current_location, new_location, county, icu, los)
        del self.readmission_date[unique_id]
        del self.readmission_location[unique_id]

    @lru_cache(maxsize=None)
    def find_icu_probability(self, age: int, size: int, current_los: int, cc: int):
        """Calculate the probability of a specific agent going to an ICU"""
        logit = -2.4035
        if age == AgeGroup.AGE1:
            logit += 0.1395
        elif age == AgeGroup.AGE2:
            logit += 0.1326
        if size > 400:
            logit += 0.1867
        if cc == 1:
            logit += 0.8169
        if current_los <= 7:
            pass
        elif current_los <= 30:
            logit += 0.2571
        else:
            logit += 0.7337
        return expit(logit)

    def change_location(self, unique_id: int, current_location: int, new_location: int, icu: bool = False):
        """Move an agent from their current_location to a new location"""
        if current_location == self.community:
            los = 0
        else:
            los = self.model.time - self.last_movement_day.get(unique_id, 0)
        self.last_movement_day[unique_id] = self.model.time
        event = (unique_id, self.model.time, current_location, new_location, int(icu), los)
        self.location.state_changes.record_event(event)
        if current_location != self.community:
            self.facilities[current_location].remove_agent(unique_id)
        self.location.previous[unique_id] = current_location
        self.location[unique_id] = new_location

    def go_home(self, unique_id: int, current_location: int):
        """Send a patient to the community"""
        self.facilities[current_location].remove_agent(unique_id)

        # Update their LOS based on how long they were actually at the facility (imporant for people who die)
        self.current_los[unique_id] = self.model.time - self.last_movement_day.get(unique_id, 0)
        self.change_location(unique_id, current_location, self.community)
        del self.leave_facility_day[unique_id]
        del self.current_los[unique_id]

    def select_los(self, new_location: int, initialize: bool = False):
        """LOS needs to be known before ICU status can be assigned. However, the ICUs could be full.
        Calculate a potential LOS (perhaps based on symptoms) to determine if an ICU will be needed.
        LOS cannot be 0. Add 1 to LOS values set to 0
        """
        use_id = new_location
        if self.facilities[new_location].category in [LocationCategories.NH.name, LocationCategories.LT.name]:
            use_id = self.facilities[new_location].category
        if initialize:
            a_list = self.initial_los_distributions[use_id]
        else:
            a_list = self.los[use_id]
        selected_los = a_list[self.model.rng.randint(0, len(a_list))]
        if selected_los == 0:
            return selected_los + 1
        return selected_los

    def assign_los(self, unique_id: int, new_location: int, initialize: bool = False, los: int = None):
        """ Given a new_location, select a LOS for a new patient """
        # ----- If going home, do nothing
        if new_location == self.community:
            return
        # ----- If LOS was pretermined, use it
        if los is not None:
            self.current_los[unique_id] = los
        else:
            self.current_los[unique_id] = self.select_los(new_location, initialize=initialize)
        self.leave_facility_day[unique_id] = self.model.time + self.current_los[unique_id]

    def move_to_nh(self, unique_id: int, current_location: int = 0):
        # Select a new NH
        county = self.model.county_code[unique_id]
        names = self.county_nh_probabilities[county]["names"]
        prob_array = self.county_nh_probabilities[county]["prob_array"]
        # Select up to X options
        count = self.params.location.NH_attempt_count
        new_locations = self.model.rng.choice(names, size=min(len(names), count), replace=False, p=prob_array)
        self.check_movement(unique_id, current_location, new_locations)

    def move_to_lt(self, unique_id: int, current_location, county: int):
        # Select a new LTACH
        names = self.county_lt_probabilities[county]["names"]
        prob_array = self.county_lt_probabilities[county]["prob_array"]
        # Select up to X options
        count = self.params.location.LT_attempt_count
        new_locations = self.model.rng.choice(names, size=min(len(names), count), replace=False, p=prob_array)
        self.check_movement(unique_id, current_location, new_locations)

    def check_movement(self, unique_id: int, current_location: int, new_locations: list):
        for new_location in new_locations:
            new_location = self.model.nodes.name_to_int[new_location]
            if self.facilities[new_location].add_agent(unique_id):
                self.assign_los(unique_id, new_location, initialize=False)
                self.change_location(unique_id, current_location, new_location)
                return
        # No homes were available - record this event for the first location tried
        failed_location = self.model.nodes.name_to_int[new_locations[0]]
        self.patients_turned_away.record_event((unique_id, self.model.time, failed_location, False))
        # Go home if neccessary
        if current_location != self.community:
            self.go_home(unique_id, current_location=current_location)

    @lru_cache(maxsize=None)
    def find_location_transitions(self, county: int, age: int, loc_int: int) -> float:
        return list(self.facility_transitions[(county, age, loc_int)])

    def add_distributions_for_transfers(self):
        """Create a CDF of hospital transfer probabilities per county, per facility.
        Each County/Facility combination should have a list of
        """
        for _, facility in self.facilities.items():
            facility.hospital_options = {}

        for county in self.model.params.counties:
            hospital_ints = self.hospitals_by_county[county]
            array = self.discharges.loc[county][hospital_ints].values
            normalized_array = normalize_and_create_cdf(array)

            for facility_int, facility in self.facilities.items():
                facility.hospital_options[county] = {}
                # Hospitals can't transfer to themselves
                if facility_int in hospital_ints:
                    temp_array = copy(array)
                    # If a county only discharges to one hospital, pick next county that uses a similar discharge rate
                    if len(temp_array) == 1:
                        row = self.discharges[facility_int]
                        temp_county = row[(0 < row) & (row < 1)].idxmax()
                        temp_hospital_ints = self.hospitals_by_county[temp_county]
                        temp_array = self.discharges.loc[temp_county][temp_hospital_ints].values
                        index = temp_hospital_ints.index(facility_int)
                        temp_array[index] = 0
                        facility.hospital_options[county]["cdf"] = normalize_and_create_cdf(temp_array)
                        facility.hospital_options[county]["ids"] = temp_hospital_ints
                        facility.hospital_options[county]["prob_array"] = normalize(temp_array)
                        continue
                    else:
                        index = hospital_ints.index(facility_int)
                        temp_array[index] = 0
                        facility.hospital_options[county]["cdf"] = normalize_and_create_cdf(temp_array)
                else:
                    facility.hospital_options[county]["cdf"] = normalized_array
                facility.hospital_options[county]["ids"] = hospital_ints
                facility.hospital_options[county]["prob_array"] = normalize(array)

    def select_facility(self, facility: Hospital, county_age_dict: dict, icu: bool):
        """Append the number of hospital patients required for a given county/age dictionary.
        This will help fill the beds with agents
        """
        rows = pd.DataFrame(self.discharges[facility.model_int])
        rows.loc[:, "Percentage"] = rows[facility.model_int] / rows[facility.model_int].sum()

        hospital_row = self.model.hospital_df.loc[facility.name]

        if icu:
            multiplier = hospital_row["ICU_NC_Agents"] / max(hospital_row["ICU_Beds"], 1)
            beds = int(round(facility.model_beds["icu_beds"] * multiplier))
        else:
            multiplier = hospital_row["Acute_NC_Agents"] / max(hospital_row["Acute_Beds"], 1)
            beds = int(round(facility.model_beds["acute_beds"] * multiplier))

        p = rows.Percentage.values
        for _ in range(0, beds):
            # --- Randomly select a county based on the percentages
            county = self.model.rng_stdlib.choices(rows.index, weights=p)[0]
            # --- Randomly select an age based on 40% <50, 20% 50-65, and 40% 65+
            options = self.model.params.age_groups
            age = self.model.rng_stdlib.choices(options, cum_weights=self.model.hospital_age_distribution)[0]
            county_age_dict[(county, age)].append(facility.model_int)

    def patients(self, category: LocationCategories):
        t1 = np.isin(self.location.values, self.model.nodes.category_ints[category])
        t2 = self.model.life.is_living
        return self.model.unique_ids[t1 & t2]

    def assign_non_nc_placeholders(self, facility):
        hospital_row = self.model.hospital_df.loc[facility.name]
        acute_non_nc = self.model.probabilistic_round(hospital_row.Acute_Non_NC_Agents * self.model.multiplier)
        placeholder_id = self.model.population.shape[0]
        for _ in range(acute_non_nc):
            facility.add_agent(placeholder_id)
        icu_non_nc = self.model.probabilistic_round(hospital_row.ICU_Non_NC_Agents * self.model.multiplier)
        for _ in range(icu_non_nc):
            facility.add_agent(placeholder_id, icu=True)


def location_dict(model, facility_transitions):
    ld = {}

    # Hospitals
    temp_ft = facility_transitions[
        ~(facility_transitions.Facility.isin([LocationCategories.LT.name, LocationCategories.NH.name]))
    ]
    for county in model.params.counties:
        for item in temp_ft.itertuples():
            key = (county, item.Age_Group, model.nodes.name_to_int[item.Facility])
            ld[key] = normalize_and_create_cdf(item[3:])

    # All LT and NH are the same
    for category in [LocationCategories.LT.name, LocationCategories.NH.name]:
        temp_ft = facility_transitions[facility_transitions.Facility == category]
        for county in model.params.counties:
            for item in temp_ft.itertuples():
                cdf = normalize_and_create_cdf(item[3:])
                for key2 in model.nodes.category_ints[category]:
                    key = (county, item.Age_Group, key2)
                    ld[key] = cdf
    return ld
