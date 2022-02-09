from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from submodels.covid19.model.state import COVIDState, VaccinationStatus
from submodels.covid19.src.cases_to_probabilities import (
    PROPORTION_OVERALL_HOSPITALIZED,
    PROPORTION_HOSPITALIZATIONS_VACC,
    PROPORTION_ICU_HOSPITALIZATIONS_VACC,
)


@pytest.mark.usefixtures("model_with_run")
class TestCOVIDInCommunity(TestCase):
    def test_action_exposures(self):
        """Make sure the number of expected modeled exposures and actual modeled exposures are close"""
        model = self.model

        cases = model.covid_cases.make_events()
        # Do not include T = 0. This is not part of the simulation.
        cases = cases[cases.Time != 0]
        cases["age"] = [model.age_group[i] for i in cases.Unique_ID]
        blocked_cases = model.blocked_covid_cases.make_events()
        blocked_cases["age"] = [model.age_group[i] for i in blocked_cases.Unique_ID]

        # ----- Test Reported Percentage
        target = 1 / model.params.case_multiplier
        value = cases.Reported.value_counts(1).loc[1]
        assert np.isclose(target, value, atol=0.01)

        # ----- Test Vaccinated Case Percentage
        target = model.params.proportion_cases_post_vaccination
        value = cases.Vacc_Status.value_counts(1).loc[VaccinationStatus.VACCINATED]
        assert np.isclose(target, value, atol=0.01)

        # ----- Test Total Cases by Age and Vaccinated Status
        input_cases = model.cases.reset_index()
        input_cases = input_cases[input_cases.Date > pd.to_datetime(model.start_date)]
        for column in [i for i in input_cases.columns if "ABM_" in i]:
            age = column.split("_")[2]
            vacc_status = column.split("_")[3]
            # Cases
            temp_cases = cases[cases.Vacc_Status == VaccinationStatus[vacc_status].value]
            temp_cases = temp_cases[temp_cases.age == int(age)]
            value = temp_cases.shape[0]
            # Blocked Cases
            if vacc_status == VaccinationStatus.VACCINATED.name:
                blocked_count = blocked_cases[blocked_cases.age == int(age)].shape[0]
                value += blocked_count
            value /= model.multiplier
            target = input_cases[column].sum()
            assert np.isclose(target / value, 1, atol=0.015)

        # ----- Test Cases by Vaccination, Reported, and Outcome
        for reported in ["reported", "nonreported"]:
            for vacc_status in VaccinationStatus:
                s = f"{reported}_{vacc_status.name.lower()}_severity"
                for age in [0, 1, 2]:
                    temp_cases = cases[cases.Vacc_Status == vacc_status]
                    if reported == "reported":
                        temp_cases = temp_cases[temp_cases.Reported]
                    else:
                        temp_cases = temp_cases[~temp_cases.Reported]
                    temp_cases = temp_cases[temp_cases.age == age]
                    vc = temp_cases["Type"].value_counts(1)
                    v1 = vc.get(COVIDState.ASYMPTOMATIC.value, 0)
                    v2 = vc.get(COVIDState.MILD.value, 0)
                    v3 = vc.get(COVIDState.SEVERE.value, 0)
                    v4 = vc.get(COVIDState.CRITICAL.value, 0)
                    values = [v1, v1 + v2, v1 + v2 + v3, v1 + v2 + v3 + v4]
                    targets = model.params.__dict__[s][age]

                    for i in range(len(values)):
                        assert np.isclose(targets[i], values[i], atol=0.03)

        # ----- Cases + Blocked Cases = Exposures
        modeled_exposures = (cases.shape[0] + blocked_cases.shape[0]) / model.multiplier
        input_cases = model.cases.reset_index()
        input_cases = input_cases[input_cases.Date > pd.to_datetime(model.start_date)]
        target_exposures = input_cases[[i for i in input_cases.columns if "ABM_" in i]].sum().sum()
        assert np.isclose(modeled_exposures / target_exposures, 1, atol=0.025)

        # ----- Blocked Cases is X% of (Blocked + Vacc Cases)
        bc = blocked_cases.shape[0]
        vc = cases[cases.Vacc_Status == VaccinationStatus.VACCINATED].shape[0]
        assert np.isclose((bc) / (vc + bc), model.params.new_vaccine_effectiveness, atol=0.01)

        # ----- Hospitalizations are correct by vaccination status
        if model.params.num_agents > 2000000:
            rc = cases[cases.Reported]
            # Severe cases
            vc = rc[rc["Type"] == COVIDState.SEVERE].Vacc_Status.value_counts(1)
            value = vc.loc[VaccinationStatus.VACCINATED]
            target = PROPORTION_HOSPITALIZATIONS_VACC
            assert np.isclose(target, value, atol=0.03)
            # Critical cases
            vc = rc[rc["Type"] == COVIDState.CRITICAL].Vacc_Status.value_counts(1)
            value = vc.loc[VaccinationStatus.VACCINATED]
            target = PROPORTION_ICU_HOSPITALIZATIONS_VACC
            assert np.isclose(target, value, atol=0.03)

        # ----- Test Overall Hospitalization
        case_outcomes = cases[cases.Reported]["Type"].value_counts(1)
        hospital_rate = case_outcomes.loc[COVIDState.CRITICAL] + case_outcomes.loc[COVIDState.SEVERE]
        assert np.isclose(hospital_rate, PROPORTION_OVERALL_HOSPITALIZED, atol=0.02)

    def test_find_exposures_in_community(self):
        # model = self.model
        pass

    def test_is_case_blocked(self):
        pass

    def test_give_covid19(self):
        pass

    def test_find_covid_severity(self):
        pass

    def test_go_to_hospital(self):
        pass
