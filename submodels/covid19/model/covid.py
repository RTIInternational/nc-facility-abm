import datetime as dt
from copy import copy
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as stats
import submodels.covid19.src.covid_data_input as cdi
from optabm.example_functions import init_distribution
from optabm.optapply import check_agents, select_agents
from optabm.state import State
from optabm.storage import EventStorage
from submodels.covid19.model.parameters import CovidParameters
from submodels.covid19.model.state import COVIDState, VaccinationStatus, WorkerType
from submodels.covid19.seir.seir import NCSEIR, seir_params
from submodels.covid19.src.healthcare_workers import (
    calculate_target_worker_counts,
    read_county_distance_data,
    read_staffing_data,
)

from model.facilities import NursingHome
from model.hospital_abm import HospitalABM
from model.state import LifeState


class COVIDModel(HospitalABM):
    def __init__(self, scenario_dir: str, run_dir: str, seed: int = 1111):
        super().__init__(scenario_dir=scenario_dir, run_dir=run_dir, seed=seed, params=CovidParameters)

        # ----- Healthcare workers
        self.assign_healthcare_workers()
        _ids = [self.healthcare_workers[i]["workers"] for i in self.healthcare_workers.keys()]
        self.hcw_ids = [j for i in _ids for j in i]

        # Setup States and Default Values
        self.covid19 = State(self, state_enum=COVIDState)
        self.vaccination_status = State(self, state_enum=VaccinationStatus)
        self.recovery_day = dict()

        # Setup Vaccinaiton
        self.vaccine_rates = self.create_vaccine_rates()

        # COVID-19 Cases by County
        self.start_date = dt.datetime.strptime(self.params.start_date, "%Y-%m-%d").date() - dt.timedelta(1)
        seir_params["current_case_multiplier"] = self.params.case_multiplier
        self.seir = NCSEIR(seir_params)
        self.cases = self.load_cases().sort_index()

        # Setup LOS options
        mu, sigma = self.params.los_mean, self.params.los_std
        self.covid_los_options = stats.truncnorm.rvs(
            (self.params.los_min - mu) / sigma, (self.params.los_max - mu) / sigma, loc=mu, scale=sigma, size=10000
        )
        self.covid_los_options = [max(int(round(i)), 1) for i in self.covid_los_options]
        self.covid_los_initial_options = init_distribution(self.rng, self.covid_los_options)

        # Setup Visitation
        self.nh_visitors = {}

        # Prepare for Initializing COVID
        self.proportion_of_cases_reported = 1 / self.params.case_multiplier
        self.covid_outcomes = [COVIDState.ASYMPTOMATIC, COVIDState.MILD, COVIDState.SEVERE, COVIDState.CRITICAL]
        self.covid_distributions = {}
        self.covid_distributions["not_reported"] = {
            VaccinationStatus.NOTVACCINATED: self.params.nonreported_notvaccinated_severity,
            VaccinationStatus.VACCINATED: self.params.nonreported_vaccinated_severity,
        }
        self.covid_distributions["reported"] = {
            VaccinationStatus.NOTVACCINATED: self.params.reported_notvaccinated_severity,
            VaccinationStatus.VACCINATED: self.params.reported_vaccinated_severity,
        }

        # Setup EventStorage
        self.covid_cases = EventStorage(column_names=["Time", "Unique_ID", "Type", "Vacc_Status", "Reported"])
        self.blocked_covid_cases = EventStorage(column_names=["Time", "Unique_ID", "County"])
        self.hcw_attendance_storage = EventStorage(
            column_names=["Time", "Unique_ID", "Location", "COVID_State", "Vacc_Status"]
        )
        self.nh_visits = EventStorage(
            column_names=["Time", "Unique_ID", "Visitor_ID", "Visitor_Order", "COVID_State", "Vacc_Status", "Location"]
        )
        self.hospitalizations = pd.DataFrame()
        self.out_of_county_cases = {}

        self.initialize_covid()

    def initialize_covid(self):
        self.assign_vaccines_to_community()
        self.assign_vaccines_to_nursing_homes()
        self.assign_healthcare_worker_vaccination()
        self.init_covid_hospitalizations()
        self.init_community_covid()

    def create_vaccine_rates(self):
        """The current vaccination data gets adjusted based on the 3 input community vaccination parameters
        Increase/decrease vaccinaiton by county based on the input parameters.
        """
        vacc_rates = cdi.vaccination_rates_by_age()
        # Adjust the vaccination rates to match the input parameter (if needed)
        vacc_rates[["adjusted_0", "adjusted_1", "adjusted_2"]] *= self.params.community_vaccination_multiplier
        return vacc_rates

    def assign_vaccines_to_community(self):
        """Each county in NC has a different vaccination rate. We want to assign vaccinations in the community
        so that counties match their vaccination rates, but also match the level of the input parameter
        Community in this context is anyone who is not living at a Nursing Home
        """
        # Select anyone not in a nursing home
        unique_ids = self.unique_ids[~np.isin(self.movement.location.values, self.nodes.category_ints["NH"])]
        # Select non healthcare workers
        unique_ids = unique_ids[~np.isin(unique_ids, self.hcw_ids)]

        # Assign vaccination by county, vaccination rate, and age
        for county in self.params.counties:
            county_str = self.county_codes_dict[county]
            county_ids = unique_ids[self.county_code[unique_ids] == county]
            county_vaccine_rates = self.vaccine_rates.loc[county_str]
            for age in self.params.age_groups:
                p = county_vaccine_rates.loc[f"adjusted_{age}"]
                age_ids = county_ids[self.age_group[county_ids] == age]
                selected_ids = check_agents(self, age_ids, p)
                self.vaccination_status[selected_ids] = VaccinationStatus.VACCINATED

    def assign_vaccines_to_nursing_homes(self):
        """Assign Vaccinations to Nursing Homes based on input parameter.
        This does not take into account county specific vaccination rates.
        """
        # Who is in a nursing homes
        unique_ids = self.unique_ids[np.isin(self.movement.location.values, self.nodes.category_ints["NH"])]
        # Select people to be vaccinated
        selected_ids = check_agents(self, unique_ids, self.params.nh_vaccination)
        self.vaccination_status[selected_ids] = VaccinationStatus.VACCINATED

    def assign_healthcare_worker_vaccination(self):
        """Assign Vaccinations to Healthcare Workers based on input parameter"""
        # Who is in a nursing homes
        unique_ids = np.array(self.hcw_ids)
        # Select people to be vaccinated
        selected_ids = check_agents(self, unique_ids, self.params.healthcare_worker_vaccination)
        self.vaccination_status[selected_ids] = VaccinationStatus.VACCINATED

    def init_covid_hospitalizations(self):
        """Given the input parameters for the number of currently hospitalized Severe and Critical patients, generate
        an equivalent amount of hospitalized patients
        """
        for k, hospitalizations in self.params.initial_hospital_cases.items():
            # Only consider agents in the commmunity who are susceptible
            f1 = self.movement.location.values == 0
            f2 = np.isin(self.covid19.values, [COVIDState.SUSCEPTIBLE.value])
            for age_group in self.params.age_groups:
                age_p = self.params.covid_hosp_age_distribution["distribution"][age_group]
                age_size = self.probabilistic_round(hospitalizations * age_p * self.multiplier)
                # Only consider agents in the
                unique_ids = self.unique_ids[f1 & f2 * (self.age_group == age_group)]
                selected_ids = select_agents(self, unique_ids, age_size)
                for unique_id in selected_ids:
                    state = COVIDState[k]
                    self.covid19.to(unique_id, state=state)
                    self.go_to_hospital(unique_id, covid_state=state, reported=True, initialize=True)

    def init_community_covid(self):
        """Setup initial COVID-19 cases in the community based on Infectious/Recovered values from SEIR model."""
        for county, county_str in self.county_codes_dict.items():
            # Total Cases
            infectious = self.cases.loc[(pd.to_datetime(self.start_date), county_str)].Infectious
            size = infectious * self.county_population[county_str] * self.multiplier
            for age, value in enumerate(self.params.covid_age_distribution["distribution"]):
                age_count = self.probabilistic_round(size * value)
                selected_ids = self.find_exposures_in_community(size=age_count, county=county, age=age)
                # All initial exposures are given covid.
                for unique_id in selected_ids:
                    severity = COVIDState.SEVERE
                    vacc_status = self.vaccination_status[unique_id]
                    # Only record mild/symptomatic. Severe/Critical are handled by other parameters.
                    while severity in [COVIDState.SEVERE, COVIDState.CRITICAL]:
                        severity, _ = self.find_covid_severity(unique_id, vacc_status)
                        if severity in [COVIDState.ASYMPTOMATIC, COVIDState.MILD]:
                            self.covid19.to(unique_id, state=severity)
                            self.recovery_day[unique_id] = self.rng_stdlib.randint(1, self.params.infection_duration)
                            continue

        # Assign Recoveries
        for county, county_str in self.county_codes_dict.items():
            # Total Cases
            recovered = self.cases.loc[(pd.to_datetime(self.start_date), county_str)].Recovered
            size = recovered * self.county_population[county_str] * self.multiplier
            for age, value in enumerate(self.params.covid_age_distribution["distribution"]):
                age_count = self.probabilistic_round(size * value)
                selected_ids = self.find_exposures_in_community(size=age_count, county=county, age=age)
                # All initial exposures are given covid.
                for unique_id in selected_ids:
                    self.covid19.to(unique_id, state=COVIDState.RECOVERED)

    def step(self):
        """The step function is the main driver of the COVID Class"""
        self.time += 1
        self.covid_step()
        self.movement.step()
        self.perform_actions()
        self.collect_data()
        self.insert_daily()
        self.save_output()

    def covid_step(self):
        self.action_recovery()
        self.action_exposures()
        self.action_nh_visitation()
        self.action_hcw_attendance()

    def action_recovery(self):
        """Sumulate recovery from COVID-19"""
        unique_ids = [key for key, value in self.recovery_day.items() if value == self.time]
        for unique_id in unique_ids:
            self.actions.append([self.covid19.to, {"unique_id": unique_id, "state": COVIDState.RECOVERED}])

    def action_exposures(self):
        """Simulate potential COVID-19 exposure in the community: One count per county, per age, per day"""
        date = self.start_date + dt.timedelta(self.time)
        cases = self.cases.loc[pd.to_datetime(date)]

        # Find agents in the community who are susceptible
        for county in self.params.counties:
            c_str = self.county_codes_dict[county]
            county_cases = cases.loc[c_str]
            for age in self.params.age_groups:
                for vacc_status in VaccinationStatus:
                    count = county_cases[f"ABM_Exposures_{age}_{vacc_status.name}"] * self.multiplier
                    count = self.probabilistic_round(count)
                    selected_ids = self.find_exposures_in_community(
                        size=count, county=county, age=age, vaccination_status=vacc_status.name
                    )
                    for unique_id in selected_ids:
                        if self._is_case_blocked(unique_id):
                            continue
                        self.actions.append([self.give_covid19, {"unique_id": unique_id}])

    def _is_case_blocked(self, unique_id: int):
        # Simulate vaccination blocking infection
        vacc_status = self.vaccination_status[unique_id]
        if vacc_status == VaccinationStatus.VACCINATED:
            # HCW:
            if unique_id in self.hcw_ids:
                if self.rng.rand() < self.params.hcw_vaccine_effectiveness:
                    self.blocked_covid_cases.record_event((self.time, unique_id, self.county_code[unique_id]))
                    return True
            # Non-HCW
            else:
                if self.rng.rand() < self.params.vaccine_effectiveness:
                    self.blocked_covid_cases.record_event((self.time, unique_id, self.county_code[unique_id]))
                    return True
        return False

    def action_nh_visitation(self):
        """ Simulate nursing home visiation """
        if self.params.include_nh_visitation:
            nh_ids = self.unique_ids[np.isin(self.movement.location.values, self.nodes.category_ints["NH"])]
            for unique_id in nh_ids:
                self.actions.append([self.nh_visitation, {"unique_id": unique_id}])

    def action_hcw_attendance(self):
        """ Simulate healthcare workers going to work """
        if self.params.include_hcw_attendance:
            for worker in self.healthcare_workers["single_site_full_time"]["workers"]:
                self.actions.append(
                    [self.hcw_attendance, {"unique_id": worker, "worker_type": WorkerType.SINGLE_SITE_FULL_TIME}]
                )
            for worker in self.healthcare_workers["single_site_part_time"]["workers"]:
                self.actions.append(
                    [self.hcw_attendance, {"unique_id": worker, "worker_type": WorkerType.SINGLE_SITE_PART_TIME}]
                )
            for worker in self.healthcare_workers["multi_site"]["workers"]:
                self.actions.append([self.hcw_attendance, {"unique_id": worker, "worker_type": WorkerType.MULTI_SITE}])
            for worker in self.healthcare_workers["contract"]["workers"]:
                self.actions.append([self.hcw_attendance, {"unique_id": worker, "worker_type": WorkerType.CONTRACT}])

    def give_covid19(self, unique_id: int):
        """If an exposure is not blocked by a person's vaccination, they are given COVID-19"""
        vacc_status = self.vaccination_status[unique_id]
        self.recovery_day[unique_id] = self.time + self.params.infection_duration
        severity, reported = self.find_covid_severity(unique_id, vacc_status)
        self.covid19.to(unique_id, state=severity)
        state = severity.value
        # Asymptomatic/Mild Cases are not hospitalized
        if state in [COVIDState.ASYMPTOMATIC, COVIDState.MILD]:
            self.covid_cases.record_event((self.time, unique_id, state, vacc_status, reported))
            return
        # Severe/Critical cases must go to a hospital
        self.go_to_hospital(unique_id, covid_state=state, reported=reported, initialize=False)

    def find_covid_severity(self, unique_id: int, vacc_status: int):
        """ Find the severity of a COVID case based on a persons age group and vaccination status"""
        if self.rng.rand() < self.proportion_of_cases_reported:
            reported = True
            distribution = self.covid_distributions["reported"]
        else:
            reported = False
            distribution = self.covid_distributions["not_reported"]
        distribution = distribution[vacc_status][self.age_group[unique_id]]
        return self.rng_stdlib.choices(self.covid_outcomes, cum_weights=distribution)[0], reported

    def go_to_hospital(self, unique_id: int, covid_state: int, reported: bool, initialize: bool = False):
        """Attempt to send a COVID agent to a hospital using the same process as non-COVID agents."""
        icu = True if covid_state == COVIDState.CRITICAL else False
        self.movement.move_to_stach(unique_id=unique_id, current_location=0, force_icu=icu)
        vacc_status = self.vaccination_status[unique_id]
        self.covid_cases.record_event((self.time, unique_id, covid_state, vacc_status, reported))

        # If patient was turned away: Do nothing - they did not make it to a hospital
        new_location = self.movement.location.values[unique_id]
        if new_location == 0:
            return
        # Overwrite their LOS with a COVID specific LOS
        if initialize:
            los = self.rng.choice(self.covid_los_initial_options)
        else:
            los = self.rng.choice(self.covid_los_options)
        self.movement.assign_los(unique_id=unique_id, new_location=new_location, los=los)
        self.recovery_day[unique_id] = self.time + los

    def find_exposures_in_community(self, size: int, county: int, age: int, vaccination_status: str = None):
        """Given a count, select people from the community who are susceptible to be potentially given COVID-19
        Args:
            size (int): The number of COVID-19 cases
            county (int, optional): A county ID if county specific unique_ids are required. Defaults to None.
            age (int): The age group to assign
        """
        if county:
            unique_ids = self._county_code_unique_ids[county]
        else:
            unique_ids = self.unique_ids
        f1 = self.movement.location.values[unique_ids] == 0
        f2 = self.covid19.values[unique_ids] == COVIDState.SUSCEPTIBLE.value
        f3 = self.age_group[unique_ids] == age
        f4 = True
        if vaccination_status == VaccinationStatus.VACCINATED.name:
            f4 = self.vaccination_status[unique_ids] == VaccinationStatus.VACCINATED
        if vaccination_status == VaccinationStatus.NOTVACCINATED.name:
            f4 = self.vaccination_status[unique_ids] == VaccinationStatus.NOTVACCINATED
        possible_ids = unique_ids[f1 & f2 & f3 & f4]
        try:
            return select_agents(self, possible_ids, size)
        except ValueError:
            # We couldn't find enough people that meet this critieria. Look at Recovered
            f2 = self.covid19.values[unique_ids] == COVIDState.RECOVERED.value
            possible_ids = unique_ids[f1 & f2 & f3 & f4]
            try:
                return select_agents(self, possible_ids, size)
            except ValueError:
                people = self.find_exposures_in_community(
                    size, county=None, age=age, vaccination_status=vaccination_status
                )
                self.out_of_county_cases[(self.time, county)] = people
                return people

    def load_cases(self):
        """Create the case forecasts using the SEIR model."""
        sd = self.start_date + dt.timedelta(1)

        df = self.seir.run_all(sd, self.params.forecast_length, self.params.r_effective)
        df = df.reset_index().rename(columns={"index": "Date"})

        if self.params.use_historical_case_counts:
            # Make sure the smooth and new cases are equal for the time period
            sc = df[df.Date >= sd]["Smooth_Cases"].sum()
            nc = df[df.Date >= sd]["New_Cases"].sum()
            df.loc[df.Date >= sd, "Smooth_Cases"] *= nc / sc
            # Multiplier cases by 1: A scaler to increase/decrease cases & 2:A case multiplier (to get infections)
            df["Est_Daily_Infections"] = (
                df.Smooth_Cases * self.params.historical_cases_multiplier * self.params.case_multiplier
            )
        df = df.set_index(["Date", "County"])
        df = self.add_abm_exposures(df)
        return df

    def add_abm_exposures(self, df):
        """Vaccinations are effective based on input parameters
        - Inflate the estimated infections to account for this.
        - Need to estimate how many exposures will be "blocked" by vaccines
        - After vaccines "block" cases in the model, we should end up with the estimated inefections.
        """
        for description in VaccinationStatus:
            for age in [0, 1, 2]:
                df[f"ABM_Exposures_{age}_{description.name}"] = np.nan
        for _, group_df in df.groupby("County"):
            for age in self.params.age_groups:
                # Estimate infections by age
                case_rate = self.params.covid_age_distribution["distribution"][age]
                age_infections = group_df.Est_Daily_Infections * case_rate
                # Estimate infections by vacc status
                vacc_infections = age_infections * self.params.proportion_cases_post_vaccination
                nonvacc_infections = age_infections * (1 - self.params.proportion_cases_post_vaccination)
                # Inflate vaccination infections to get "exposures"
                vacc_exposures = vacc_infections / (1 - self.params.baseline_vaccine_effectiveness)

                # For Vaccinations - infections are inflated to get exposures.
                df.loc[group_df.index, f"ABM_Exposures_{age}_{VaccinationStatus.VACCINATED.name}"] = vacc_exposures
                # For nonvaccination - Infections = Exposures
                non_vacc_name = f"ABM_Exposures_{age}_{VaccinationStatus.NOTVACCINATED.name}"
                df.loc[group_df.index, non_vacc_name] = nonvacc_infections

        return df

    def nh_visitation(self, unique_id):
        """Simulate NH Visitation: A visitor must pass several barriers before visiting"""
        self.assign_nh_visitors(unique_id)

        for visitor, (visitor_id, probability) in enumerate(self.nh_visitors[unique_id].items()):
            ""
            visitor += 1
            # 0: Simulate random probability of visiting
            if self.rng.rand() > probability:
                continue
            # 1: Must be in the community
            if self.movement.location[visitor_id] != 0:
                continue
            # 2: Must be alive
            if self.life[visitor_id] == LifeState.DEAD:
                continue
            # 3: If "MILD" symptoms, they may choose to stay home
            covid_state = self.covid19[visitor_id]
            if covid_state == COVIDState.MILD:
                if self.rng.rand() < self.params.visitors_with_mild_who_stay_home:
                    continue
            # 4: Cannot have Severe/Critical Symptoms
            if covid_state in [COVIDState.SEVERE, COVIDState.CRITICAL]:
                continue

            vaccination_status = self.vaccination_status[visitor_id]
            location = self.movement.location[unique_id]

            self.nh_visits.record_event(
                (self.time, unique_id, visitor_id, visitor, covid_state, vaccination_status, location)
            )

    def assign_nh_visitors(self, unique_id):
        """Each NH resident gets a list of visitors. We first select the number, and then the agents.
        Visitors are also selected by age.
        """
        if unique_id in self.nh_visitors:
            return
        cdf = self.params.visitation_distribution["distribution"]
        count = self.rng_stdlib.choices(self.params.visitation_distribution["count"], cum_weights=cdf)[0]

        self.nh_visitors[unique_id] = {}
        if count > 0:
            unique_ids = self._county_code_unique_ids[self.county_code[unique_id]]
            # Find out how many people from each age group are required
            age_groups = []
            for visitor_number in range(1, count + 1):
                cdf = self.params.visitor_age_distribution[visitor_number]
                age_groups.append(self.rng_stdlib.choices(self.params.age_groups, cum_weights=cdf)[0])
            # Select the number needed from each age group, keeping track of the order
            # This prevents the same ID from being selected twice, and reduces filter time
            selected_ids = [-1] * count
            for age_group in set(age_groups):
                locations = [i for i, v in enumerate(age_groups) if v == age_group]
                age_unique_ids = unique_ids[self.age_group[unique_ids] == age_group]
                s_ids = select_agents(self, age_unique_ids, age_groups.count(age_group))
                for i, v in enumerate(locations):
                    selected_ids[v] = s_ids[i]

            for i in range(count):
                p = self.params.visitor_frequency_distribution[i + 1] / 30
                self.nh_visitors[unique_id][selected_ids[i]] = p

    def assign_healthcare_workers(self):
        """
        Assign healthcare workers to the facilities at which they'll work.

        Designates a subset of the model's agents as healthcare workers. Healthcare workers are divided into
        various types:

        - Single site full time employees
        - Single site part time employees
        - Multi site employees
        - Contract workers

        Each county must have a certain number of employees, so employees are allocated by county. Contract workers
        are selected at random and assumed to follow population density patterns.

        Once workers are created, they are assigned to facilities where they will work. Different worker types
        are assigned differently.

        Currently only applies to nursing homes.
        """
        self.healthcare_workers = {
            "single_site_full_time": {"workers": [], "assignments": {}},
            "single_site_part_time": {"workers": [], "assignments": {}},
            "multi_site": {"workers": [], "assignments": {}, "secondary_site_assigned": []},
            "contract": {"workers": [], "assignments": {}},
        }

        nursing_homes = [facility for facility in self.nodes.facilities.values() if type(facility) == NursingHome]

        worker_counts = calculate_target_worker_counts(read_staffing_data(), self.params, self.multiplier)
        worker_counts = worker_counts.set_index("federal_provider_number")
        county_codes = worker_counts.County_Code.unique().tolist()

        # Assign employees by county

        for county_code in county_codes:
            # First make the employees
            county_worker_counts = worker_counts[worker_counts.County_Code.eq(county_code)].copy()
            county_nursing_homes = [nh for nh in nursing_homes if nh.county_code == county_code]
            county_agents = self._county_code_unique_ids[county_code]

            n_single_site_full_time_workers = int(county_worker_counts.target_single_site_full_time_workers.sum())
            n_single_site_part_time_workers = int(county_worker_counts.target_single_site_part_time_workers.sum())
            n_multi_site_workers = int(county_worker_counts.target_multi_site_primary_workers.sum())

            county_single_site_full_time_workers = select_agents(self, county_agents, n_single_site_full_time_workers)
            county_agents = county_agents[~np.isin(county_agents, county_single_site_full_time_workers)]
            county_single_site_part_time_workers = select_agents(self, county_agents, n_single_site_part_time_workers)
            county_agents = county_agents[~np.isin(county_agents, county_single_site_part_time_workers)]
            county_multi_site_workers = select_agents(self, county_agents, n_multi_site_workers)

            self.healthcare_workers["single_site_full_time"]["workers"].extend(
                list(county_single_site_full_time_workers)
            )
            self.healthcare_workers["single_site_part_time"]["workers"].extend(
                list(county_single_site_part_time_workers)
            )
            self.healthcare_workers["multi_site"]["workers"].extend(list(county_multi_site_workers))
            # Now assign them to facilities
            if len(county_nursing_homes) > 1:
                self.assign_healthcare_workers_to_n_sites(
                    county_worker_counts,
                    county_nursing_homes,
                    county_single_site_full_time_workers,
                    county_single_site_part_time_workers,
                    county_multi_site_workers,
                )
            elif len(county_nursing_homes) == 1:
                self.assign_healthcare_workers_to_one_site(
                    county_nursing_homes,
                    county_single_site_full_time_workers,
                    county_single_site_part_time_workers,
                    county_multi_site_workers,
                )
            else:
                raise ValueError(f"No sites found for county {county_code}, expected one or more")

        # Assign remaining secondary sites to multi site employees

        county_distances = read_county_distance_data(counties_to_include=county_codes)
        county_distances = county_distances.groupby("county_from")

        self.assign_remaining_secondary_sites(county_codes, county_distances, nursing_homes, worker_counts)

        # Select agents to become contract workers. It's more difficult to figure out exactly how many should be in
        # each county, so we just select at random and let population density do it's thing.

        county_sums = worker_counts.groupby("County_Code").sum().reset_index()
        counties_with_10_contract_workers = county_sums.query("target_contract_workers >= 10").County_Code.tolist()

        n_contract_workers = int(worker_counts.target_contract_workers.sum() / 3)
        employees = (
            self.healthcare_workers["single_site_full_time"]["workers"]
            + self.healthcare_workers["single_site_part_time"]["workers"]
            + self.healthcare_workers["multi_site"]["workers"]
        )
        employees_array = np.array(employees)
        county_filter = np.isin(self.county_code, counties_with_10_contract_workers)
        employee_filter = ~np.isin(self.unique_ids, employees_array)
        free_agents = self.unique_ids[county_filter & employee_filter]
        self.healthcare_workers["contract"]["workers"] = select_agents(self, free_agents, n_contract_workers)

        # Assign contract workers

        self.assign_contract_workers(county_codes, county_distances, nursing_homes, worker_counts)

    def assign_healthcare_workers_to_n_sites(
        self,
        county_worker_counts: pd.DataFrame,
        county_nursing_homes: List[NursingHome],
        county_single_site_full_time_workers: List,
        county_single_site_part_time_workers: List,
        county_multi_site_workers: List,
    ):
        """
        Does the heavy lifting of assigning workers in counties with multiple sites.

        Single-site workers are assigned first, by shuffling the list of workers and assigning them in chunks,
        based on the target number of staff, as we iterate through sites.

        Multi-site workers are assigned their primary site in a similar fashion as single-site workers.

        The secondary site is assigned for each multi-site worker by iterating through sites in need of secondary
        workers until one is found which is not the same as their primary site. This avoids the possibility of a
        worker having the same site assigned as primary and secondary if we had used the same shuffling approach here.
        """

        # Single Site

        # Full Time
        self.rng.shuffle(county_single_site_full_time_workers)
        workers_assigned = 0

        for site in county_nursing_homes:
            v = county_worker_counts.loc[site.federal_provider_number, "target_single_site_full_time_workers"]
            target_single_site_full_time_workers = int(v)
            for worker in county_single_site_full_time_workers[
                workers_assigned : workers_assigned + target_single_site_full_time_workers
            ]:
                site.single_site_full_time_workers.append(worker)
                self.healthcare_workers["single_site_full_time"]["assignments"][worker] = site
            workers_assigned += target_single_site_full_time_workers

        # Part Time
        self.rng.shuffle(county_single_site_part_time_workers)
        workers_assigned = 0

        for site in county_nursing_homes:
            v = county_worker_counts.loc[site.federal_provider_number, "target_single_site_part_time_workers"]
            target_single_site_part_time_workers = int(v)
            for worker in county_single_site_part_time_workers[
                workers_assigned : workers_assigned + target_single_site_part_time_workers
            ]:
                site.single_site_part_time_workers.append(worker)
                self.healthcare_workers["single_site_part_time"]["assignments"][worker] = site
            workers_assigned += target_single_site_part_time_workers

        # Multi Site

        # Primary
        self.rng.shuffle(county_multi_site_workers)
        workers_assigned = 0

        for site in county_nursing_homes:
            v = county_worker_counts.loc[site.federal_provider_number, "target_multi_site_primary_workers"]
            target_multi_site_primary_workers = int(v)
            for worker in county_multi_site_workers[
                workers_assigned : workers_assigned + target_multi_site_primary_workers
            ]:
                site.multi_site_primary_workers.append(worker)
                self.healthcare_workers["multi_site"]["assignments"][worker] = {"primary": site}
            workers_assigned += target_multi_site_primary_workers

        # Secondary
        self.rng.shuffle(county_multi_site_workers)

        for site in county_nursing_homes:
            v = county_worker_counts.loc[site.federal_provider_number, "target_multi_site_secondary_workers"]
            for _ in range(int(v)):
                for worker in county_multi_site_workers:
                    f1 = worker not in site.multi_site_primary_workers
                    f2 = worker not in self.healthcare_workers["multi_site"]["secondary_site_assigned"]
                    if f1 & f2:
                        site.multi_site_secondary_workers.append(worker)
                        self.healthcare_workers["multi_site"]["secondary_site_assigned"].append(worker)
                        self.healthcare_workers["multi_site"]["assignments"][worker]["secondary"] = site
                        break

    def assign_healthcare_workers_to_one_site(
        self,
        county_nursing_homes: List[NursingHome],
        county_single_site_full_time_workers: List,
        county_single_site_part_time_workers: List,
        county_multi_site_workers: List,
    ):
        """
        In counties with only one site, we only assign workers to their primary site.

        Multi-site workers will need to be assigned a secondary site from another county later, since there are no
        other options in this county.
        """
        assert len(county_nursing_homes) == 1, "Assign workers to one site called mistakenly when county has >1 sites"

        site = county_nursing_homes[0]

        for worker in county_single_site_full_time_workers:
            site.single_site_full_time_workers.append(worker)
            self.healthcare_workers["single_site_full_time"]["assignments"][worker] = site

        for worker in county_single_site_part_time_workers:
            site.single_site_part_time_workers.append(worker)
            self.healthcare_workers["single_site_part_time"]["assignments"][worker] = site

        for worker in county_multi_site_workers:
            site.multi_site_primary_workers.append(worker)
            self.healthcare_workers["multi_site"]["assignments"][worker] = {"primary": site}

    def assign_remaining_secondary_sites(
        self,
        county_codes: list,
        county_distances: pd.DataFrame,
        nursing_homes: List[NursingHome],
        worker_counts: pd.DataFrame,
    ):
        """
        Iterate through all sites in need of secondary workers. For each site needing N secondary workers, find the N
        closest multi-site workers in need of a secondary site and assign them.

        Shuffle order of counties so county alphabetical order does not influence likelihood of having faraway
        secondary workers. If we don't do this, sites in counties later in alphabetical order are more likely to have
        faraway workers, since the pool of available workers get smaller as we iterate through counties.
        """
        unassigned_worker_counties = {
            worker: self.county_code[worker]
            for worker in self.healthcare_workers["multi_site"]["workers"]
            if worker not in self.healthcare_workers["multi_site"]["secondary_site_assigned"]
        }
        shuffled_codes = copy(county_codes)
        self.rng.shuffle(shuffled_codes)

        for county_code in shuffled_codes:
            neighbors = (
                county_distances.get_group(county_code).sort_values("distance", ascending=True).county_to.tolist()
            )
            available_workers = []
            for neighbor in neighbors + [county_code]:
                for worker in unassigned_worker_counties.keys():
                    if unassigned_worker_counties[worker] == neighbor:
                        available_workers.append(worker)

            county_nursing_homes = [nh for nh in nursing_homes if nh.county_code == county_code]
            self.rng.shuffle(county_nursing_homes)

            for site in county_nursing_homes:
                v = worker_counts.loc[site.federal_provider_number, "target_multi_site_secondary_workers"]
                secondary_workers_needed = int(v) - len(site.multi_site_secondary_workers)
                if secondary_workers_needed > 0:
                    if len(available_workers) < secondary_workers_needed:
                        print(
                            f"Not enough available workers ({len(available_workers)}) to fulfill demand ({secondary_workers_needed}) for site {site} in county code {county_code}"
                        )
                    else:
                        for _ in range(secondary_workers_needed):
                            worker = available_workers[0]
                            site.multi_site_secondary_workers.append(worker)
                            available_workers.pop(0)
                            unassigned_worker_counties.pop(worker)
                            self.healthcare_workers["multi_site"]["secondary_site_assigned"].append(worker)
                            self.healthcare_workers["multi_site"]["assignments"][worker]["secondary"] = site

    def assign_contract_workers(
        self,
        county_codes: list,
        county_distances: pd.DataFrame,
        nursing_homes: List[NursingHome],
        worker_counts: pd.DataFrame,
    ):
        """
        Iterate through all sites. For each site needing N contract workers, find the N closest contract workers in
        need of a site and assign them.

        Shuffle order of counties so county alphabetical order does not influence likelihood of having faraway contract
        workers. If we don't do this, sites in counties later in alphabetical order are more likely to have faraway
        workers, since the pool of available workers get smaller as we iterate through counties.
        """
        county_workers = {}
        for county in county_codes:
            county_ids = self.healthcare_workers["contract"]["workers"][
                self.county_code[self.healthcare_workers["contract"]["workers"]] == county
            ]
            county_workers[county] = {worker: 0 for worker in county_ids}
        assert sum([len(i) for i in county_workers.values()]) == len(self.healthcare_workers["contract"]["workers"])

        shuffled_codes = copy(county_codes)
        self.rng.shuffle(shuffled_codes)

        for county_code in shuffled_codes:
            neighbors = (
                county_distances.get_group(county_code).sort_values("distance", ascending=True).county_to.tolist()
            )
            county_nursing_homes = [nh for nh in nursing_homes if nh.county_code == county_code]

            for site in county_nursing_homes:
                v = worker_counts.loc[site.federal_provider_number, "target_contract_workers"]
                contract_workers_needed = int(v)

                if contract_workers_needed > 0:

                    def find_available_workers():
                        available_workers = []
                        # Add county first: sort so workers with fewer sites assigned will be assigned sites first
                        sorted_county_workers = sorted(county_workers[county_code], key=county_workers[county_code].get)
                        available_workers.extend(sorted_county_workers)
                        # Neighbors second
                        for neighbor in neighbors:
                            sorted_neighbor_workers = sorted(county_workers[neighbor], key=county_workers[neighbor].get)
                            available_workers.extend(sorted_neighbor_workers)
                        # Nonneighbors third
                        for nonneighbor in [i for i in county_codes if i not in neighbors + [county_code]]:
                            sorted_workers = sorted(county_workers[nonneighbor], key=county_workers[nonneighbor].get)
                            available_workers.extend(sorted_workers)
                        return available_workers

                    available_workers = find_available_workers()

                    for _ in range(contract_workers_needed):
                        # It's possible that the final site or two will have multiple openings, but only 1 available
                        # worker. If this happens, we assign that worker to the site twice.
                        try:
                            worker = available_workers[0]
                        except IndexError:
                            available_workers = find_available_workers()
                            worker = available_workers[0]
                        county = self.county_code[worker]
                        site.contract_workers.append(worker)
                        county_workers[county][worker] += 1
                        if worker in self.healthcare_workers["contract"]["assignments"].keys():
                            self.healthcare_workers["contract"]["assignments"][worker].append(site)
                        else:
                            self.healthcare_workers["contract"]["assignments"][worker] = [site]
                        if county_workers[county][worker] == 3:
                            county_workers[county].pop(worker)
                        available_workers.pop(0)

    def hcw_attendance(self, unique_id: int, worker_type: WorkerType):
        # This block of conditions applies to all worker types

        # 1: Must be in the community
        if self.movement.location[unique_id] != 0:
            return
        # 2: Must be alive
        if self.life[unique_id] == LifeState.DEAD:
            return
        # 3: If "MILD" symptoms, they may choose to stay home
        worker_state = self.covid19[unique_id]
        if worker_state == COVIDState.MILD:
            if self.rng.rand() < self.params.hcw_with_mild_who_stay_home:
                return
        # 4: Cannot have Severe/Critical Symptoms
        elif worker_state in [COVIDState.SEVERE, COVIDState.CRITICAL]:
            return

        # The final condition - whether today is a workday or not - varies by worker type
        vacc_status = self.vaccination_status[unique_id]
        if worker_type == WorkerType.SINGLE_SITE_FULL_TIME:
            # Simulate random probability that today is a workday for this worker
            if self.rng.random() > self.params.workday_prob:
                return

            # At this point, the worker has made it through all conditions, so they go to work today.
            site = self.healthcare_workers["single_site_full_time"]["assignments"][unique_id]
            self.hcw_attendance_storage.record_event((self.time, unique_id, site.model_int, worker_state, vacc_status))

        elif worker_type == WorkerType.SINGLE_SITE_PART_TIME:
            # Simulate random probability that today is a workday for this worker.
            # Assume that part time workers still work full days, they just work fewer of them.
            if self.rng.random() > self.params.workday_prob * self.params.part_time:
                return

            # At this point, the worker has made it through all conditions, so they go to work today.
            site = self.healthcare_workers["single_site_part_time"]["assignments"][unique_id]
            self.hcw_attendance_storage.record_event((self.time, unique_id, site.model_int, worker_state, vacc_status))

        elif worker_type == WorkerType.MULTI_SITE:
            # Simulate random probability that today is a workday for this worker
            if self.rng.random() > self.params.workday_prob:
                return

            # At this point, the worker has made it through all conditions, so they go to work today.
            # Since this is a multi-site worker, compare a random draw to determine whether they
            # work at their primary or secondary site.
            # NOTE: It is possible a multi site worker only has one site.
            f1 = self.rng.random() < self.params.pct_time_second_site
            if (len(self.healthcare_workers["multi_site"]["assignments"][unique_id]) == 2) & f1:
                site = self.healthcare_workers["multi_site"]["assignments"][unique_id]["secondary"]
            else:
                site = self.healthcare_workers["multi_site"]["assignments"][unique_id]["primary"]
            self.hcw_attendance_storage.record_event((self.time, unique_id, site.model_int, worker_state, vacc_status))

        elif worker_type == WorkerType.CONTRACT:
            # Simulate random probability that today is a workday for this worker
            if self.rng.random() > self.params.workday_prob:
                return

            # At this point, the worker has made it through all conditions, so they go to work today.
            # Since this is a contract worker, randomly choose one of their sites.
            site = self.rng.choice(self.healthcare_workers["contract"]["assignments"][unique_id])
            self.hcw_attendance_storage.record_event((self.time, unique_id, site.model_int, worker_state, vacc_status))

    def collect_data(self):
        """Different parameters will cause the model to collect different pieces of information"""
        if self.params.track_hospitalizations:
            # Limit to only movement events from today
            events = self.movement.location.state_changes.make_events()
            events = events[(events.Time == self.time) & (events.Location == 0)].copy()
            # Replace LOS
            events["LOS"] = [self.movement.leave_facility_day[i] - self.time for i in events.Unique_ID]
            events["Age"] = [self.age_group[i] for i in events.Unique_ID]
            events["County"] = [self.county_code[i] for i in events.Unique_ID]
            events["Comorbidities"] = [self.concurrent_conditions[i] for i in events.Unique_ID]
            # Split Events into Categories & Append the COVID-19 Status
            events["Category"] = [self.nodes.facilities[i].category for i in events.New_Location]
            events["COVID_Status"] = [self.covid19[i] for i in events.Unique_ID]
            events["Vaccination_Status"] = [self.vaccination_status[i] for i in events.Unique_ID]
            self.hospitalizations = pd.concat([self.hospitalizations, events])

    def save_output(self):
        if self.time != self.params.time_horizon:
            return
        # Save All Cases
        self.covid_cases.make_events().to_csv(self.output_dir.joinpath("covid_cases.csv"), index=False)
        # Save Hospitalizations
        if self.params.track_hospitalizations:
            self.hospitalizations.to_csv(self.output_dir.joinpath("hospitalizations.csv"), index=False)
        # Save HCW Attendance
        if self.params.include_hcw_attendance:
            hcw_df = self.hcw_attendance_storage.make_events()
            hcw_df.to_csv(self.output_dir.joinpath("hcw_attendance.csv"), index=False)
            hcw_df = hcw_df.groupby(["Location", "Time", "COVID_State"]).size()
            hcw_df = hcw_df.reset_index()
            hcw_df.columns = ["Location", "Time", "COVID_State", "Count"]
            # hcw_df['Location'] = [self.nodes.facilities[i].federal_provider_number for i in hcw_df.Location.values]
            hcw_df.to_csv(self.output_dir.joinpath("hcw_attendance_counts.csv"), index=False)
        # Save NH Visits
        if self.params.simulate_nh_visitation:
            nhv_df = self.nh_visits.make_events()
            nhv_df.to_csv(self.output_dir.joinpath("nh_visits.csv"), index=False)
