from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from submodels.covid19.model.state import COVIDState, VaccinationStatus


@pytest.mark.usefixtures("model")
class TestInitiation(TestCase):
    def test_assign_vaccines_to_community_by_age(self):
        """Each age group has a specific vaccination rate"""
        model = self.model
        # Only considered non-NH people
        unique_ids = model.unique_ids[~np.isin(model.movement.location.values, model.nodes.category_ints["NH"])]

        # Remove healthcare workers
        unique_ids = unique_ids[~np.isin(unique_ids, model.hcw_ids)]

        for age in model.params.age_groups:
            # Find the target
            vacccines = (model.vaccine_rates[f"pop_{age}"] * model.vaccine_rates[f"adjusted_{age}"]).sum()
            population = model.vaccine_rates[f"pop_{age}"].sum()
            target = vacccines / population

            # Find the value
            age_ids = unique_ids[model.age_group[unique_ids] == age]
            vacc_status = pd.Series(model.vaccination_status[age_ids]).value_counts(1)
            value = vacc_status[VaccinationStatus.VACCINATED]
            assert np.isclose(value, target, atol=0.01)

    def test_assign_vaccines_to_community_by_county(self):
        """Counties should have vaccination rates in the order or the real data"""
        model = self.model
        vaccine_rates = model.vaccine_rates.copy()

        # Only considered non-NH people
        unique_ids = model.unique_ids[~np.isin(model.movement.location.values, model.nodes.category_ints["NH"])]

        county_rates = {}
        for county in model.params.counties:
            county_ids = unique_ids[model.county_code[unique_ids] == county]
            vacc_rate = pd.Series(model.vaccination_status[county_ids]).value_counts(1)[VaccinationStatus.VACCINATED]
            county_str = model.county_codes_dict[county]
            county_rates[county_str] = vacc_rate

        vaccine_rates["Model"] = vaccine_rates.index.map(county_rates)
        vaccine_rates["Expected"] = 0
        population = vaccine_rates[["pop_0", "pop_1", "pop_2"]].sum(axis=1)
        for age in [0, 1, 2]:
            vaccine_rates["Expected"] += vaccine_rates[f"adjusted_{age}"] * vaccine_rates[f"pop_{age}"] / population

        # ----- On averages, counties should have vaccination rates close to the input data
        assert np.isclose((vaccine_rates.Model / vaccine_rates.Expected).mean(), 1, atol=0.01)

    def test_assign_vaccines_to_nursing_homes(self):
        model = self.model
        nh_ids = model.unique_ids[np.isin(model.movement.location.values, model.nodes.category_ints["NH"])]
        target = model.params.nh_vaccination
        value = pd.Series(model.vaccination_status[nh_ids]).value_counts(1).loc[VaccinationStatus.VACCINATED]
        assert np.isclose(target, value, atol=0.03)

    def test_assign_healthcare_worker_vaccination(self):
        model = self.model
        hcw_ids = model.hcw_ids
        target = model.params.healthcare_worker_vaccination
        values = [model.vaccination_status[i] for i in hcw_ids]
        value = pd.Series(values).value_counts(1).loc[VaccinationStatus.VACCINATED]
        assert np.isclose(target, value, atol=0.03)

    def test_init_covid_hospitalizations(self):
        """Make sure SEVERE and CRITICAL cases start as intended"""
        model = self.model
        counts = model.covid19.value_counts()

        # Severe
        value = counts[COVIDState.SEVERE.name]
        target = model.params.initial_hospital_cases[COVIDState.SEVERE.name] * model.multiplier
        assert np.isclose(value / target, 1, atol=0.05)
        # Crtical
        value = counts[COVIDState.CRITICAL.name]
        target = model.params.initial_hospital_cases[COVIDState.CRITICAL.name] * model.multiplier
        assert np.isclose(value / target, 1, atol=0.1)

    def test_init_community_infections(self):
        """The infectious rate in the community should match the SEIR output"""
        model = self.model

        # ----- The current number of infections should have been initiated
        counts = model.covid19.value_counts()
        value = counts[COVIDState.MILD.name] + counts[COVIDState.ASYMPTOMATIC.name]
        target = 0
        county_dict = {v: k for k, v in model.county_codes_dict.items()}
        for county_str, row in model.cases.loc[pd.to_datetime(model.start_date)].iterrows():
            county = county_dict[county_str]
            target += len(model._county_code_unique_ids[county]) * row.Infectious
        assert np.isclose(value / target, 1, atol=0.02)

        # ----- The current community infections should match the input parameters for age
        ids = model.unique_ids[np.isin(model.covid19.values, [COVIDState.MILD, COVIDState.ASYMPTOMATIC])]
        age_rates = pd.Series(model.age_group[ids]).value_counts(1)
        for age in model.params.age_groups:
            target = model.params.covid_age_distribution["distribution"][age]
            value = age_rates.loc[age]
            assert np.isclose(target, value, atol=0.01)

        # TODO The split of MILD Assymptomatic should match the input parameters

    def test_init_community_recovered(self):
        """The infectious rate in the community should match the SEIR output"""
        model = self.model

        # ----- The current number of infections should have been initiated
        counts = model.covid19.value_counts()
        value = counts[COVIDState.RECOVERED.name]
        target = 0
        county_dict = {v: k for k, v in model.county_codes_dict.items()}
        for county_str, row in model.cases.loc[pd.to_datetime(model.start_date)].iterrows():
            county = county_dict[county_str]
            target += len(model._county_code_unique_ids[county]) * row.Recovered
        assert np.isclose(value / target, 1, atol=0.02)

    def test_create_vaccine_rates(self):
        pass
