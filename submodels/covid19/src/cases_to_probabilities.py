from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd
from sodapy import Socrata
from submodels.covid19.model.state import COVIDState

from submodels.covid19.model.parameters import CovidParameters


def calculate_parameter_setup():
    # Calculate the percent of post-vaccination COVID cases that end up in a hospital
    vacc_hosp_rate = (PROPORTION_HOSPITALIZATIONS_VACC * PROPORTION_OVERALL_HOSPITALIZED) / PROPORTION_CASES_ARE_VACC
    vacc_hosp_severe = vacc_hosp_rate * (1 - PROPORTION_VACC_IN_ICU)
    vacc_hosp_critival = vacc_hosp_rate * PROPORTION_VACC_IN_ICU
    # Create template COVIDStatus by Vaccination Status Dictionary
    vacc_params = OrderedDict()
    vacc_params[COVIDState.ASYMPTOMATIC] = PROPORTION_ASYMPTOMATIC_VACCINATED
    vacc_params[COVIDState.MILD] = 1 - vacc_params[COVIDState.ASYMPTOMATIC] - vacc_hosp_rate
    vacc_params[COVIDState.SEVERE] = vacc_hosp_severe
    vacc_params[COVIDState.CRITICAL] = vacc_hosp_critival
    assert sum([i for i in vacc_params.values()]) == 1

    # Calculate the percent of non-vaccinated COVID cases that end up in a hospital
    nonvacc_hosp_rate = (
        PROPORTION_HOSPITAL_CASES_NONVACC * PROPORTION_OVERALL_HOSPITALIZED
    ) / PROPORTION_CASES_ARE_NONVACC
    nonvacc_hosp_severe = nonvacc_hosp_rate * (1 - PROPORTION_NONVACC_IN_ICU)
    nonvacc_hosp_critival = nonvacc_hosp_rate * PROPORTION_NONVACC_IN_ICU
    # Not Vaccinated
    nonvacc_params = OrderedDict()
    nonvacc_params[COVIDState.ASYMPTOMATIC] = PROPORTION_ASYMPTOMATIC_NONVACCINATED
    nonvacc_params[COVIDState.MILD] = 1 - nonvacc_params[COVIDState.ASYMPTOMATIC] - nonvacc_hosp_rate
    nonvacc_params[COVIDState.SEVERE] = nonvacc_hosp_severe
    nonvacc_params[COVIDState.CRITICAL] = nonvacc_hosp_critival
    assert sum([i for i in nonvacc_params.values()]) == 1
    return vacc_params, nonvacc_params


# https://covid19.ncdhhs.gov/media/380/open on 01/24/2022. Updated by DHHS on 01/20
# ----- Page 13
PROPORTION_HOSPITALIZATIONS_VACC = 0.2836
PROPORTION_ICU_HOSPITALIZATIONS_VACC = 0.1709
PROPORTION_HOSPITAL_CASES_NONVACC = 1 - PROPORTION_HOSPITALIZATIONS_VACC
# (.1706 * 742) / (.2836 * 4225)
PROPORTION_VACC_IN_ICU = 0.106  # ((% vacc in ICU w/ COVID) * N ICU COVID ) / ((% vacc hosp w/ COVID) * N hosp COVID)
# (.8294 * 742) / ((1 - .2836) * 4225)
PROPORTION_NONVACC_IN_ICU = 0.203  # ((% non-vacc in ICU w/ COVID) * N ICU COVID ) / ((% non-vacc w/ COVID) * N COVID)
# ----- Page 12
PROPORTION_CASES_ARE_VACC = CovidParameters().proportion_cases_post_vaccination
PROPORTION_CASES_ARE_NONVACC = 1 - PROPORTION_CASES_ARE_VACC
# Admissions / Reported Cases for 12/15-1/13
PROPORTION_OVERALL_HOSPITALIZED = 0.02815

PROPORTION_ASYMPTOMATIC_VACCINATED = 0.25  #: https://pubmed.ncbi.nlm.nih.gov/34437521/
PROPORTION_ASYMPTOMATIC_NONVACCINATED = 0.05

vacc_params, nonvacc_params = calculate_parameter_setup()
PROPORTION_VACCINATED_HOSPITALIZED = vacc_params[COVIDState.SEVERE] + vacc_params[COVIDState.CRITICAL]
PROPORTION_NONVACCINATED_HOSPITALIZED = nonvacc_params[COVIDState.SEVERE] + nonvacc_params[COVIDState.CRITICAL]


def prepare_data() -> pd.DataFrame:
    """Download the latest data from the CDC on reported COVID-19 Cases for North Carolain
        - Filter to June 2021 and later
        - Fix the age group
        - Filter missing data

    Returns:
        pd.DataFrame: [description]
    """
    client = Socrata("data.cdc.gov", None)

    results = client.get("n8mc-b4w4", res_state="NC", limit=500_000)
    cdc_df = pd.DataFrame.from_records(results)

    ym_filter = (cdc_df.case_month.str[0:4] == "2021") & (cdc_df.case_month.str[5:].astype(int) > 5)
    cdc_df = cdc_df[ym_filter].copy()

    age_map = {"0 - 17 years": 0, "18 to 49 years": 0, "50 to 64 years": 1, "65+ years": 2}
    cdc_df.age_group = cdc_df.age_group.map(age_map)
    cdc_df = cdc_df[~cdc_df.age_group.isna()].copy()

    cdc_df = cdc_df[cdc_df.hosp_yn.isin(["No", "Yes"])]

    return cdc_df


def make_case_proportions(cases_by_age: pd.Series) -> Tuple[dict, dict]:
    """Given informaiton on:
        - the distribution of cases by age
        - & the proportion of cases that are from vaccinated individuals

    Calculate the proportion of all cases that are vaccinated or unvaccinated by age

    Args:
        cases_by_age (pd.Series): [description]

    Returns:
        Tuple[dict, dict]: [description]
    """
    vacc_case_proportions = {}
    nonvacc_case_proportions = {}
    for age_group in [0, 1, 2]:
        age_group_cases = cases_by_age[age_group]
        vacc_case_proportions[age_group] = age_group_cases * PROPORTION_CASES_ARE_VACC
        nonvacc_case_proportions[age_group] = age_group_cases * (1 - PROPORTION_CASES_ARE_VACC)

    assert sum(vacc_case_proportions.values()) + sum(nonvacc_case_proportions.values()) == 1

    return vacc_case_proportions, nonvacc_case_proportions


def create_age_df(age_df: pd.DataFrame, vacc_case_proportions: dict, nonvacc_case_proportions: dict) -> pd.DataFrame:
    """Calculate the probability that a (reported) vaccinated or (reported) unvaccinated case will be Severe or Critical

    This occurs in 2 steps:
        1. Find the overall hospitalization probability
        2. Find the probability of Severe (Non-ICU) or Critical (ICU) cases

    Args:
        age_df (pd.DataFrame): [description]
        vacc_case_proportions (dict): [description]
        nonvacc_case_proportions (dict): [description]

    Returns:
        pd.DataFrame: [description]
    """
    # STEP 1: Probability of a case being hospitalized by vaccination status

    # Add Proportion of all Cases by age
    age_df["P_of_All_Cases(Vacc)"] = [vacc_case_proportions[i] for i in age_df.index]
    age_df["P_of_All_Cases(NonVacc)"] = [nonvacc_case_proportions[i] for i in age_df.index]
    assert age_df["P_of_All_Cases(Vacc)"].sum() + age_df["P_of_All_Cases(NonVacc)"].sum() == 1
    assert np.isclose(age_df["P_of_All_Cases(Vacc)"].sum() / PROPORTION_CASES_ARE_VACC, 1, atol=0.00001)
    assert np.isclose(age_df["P_of_All_Cases(NonVacc)"].sum() / (1 - PROPORTION_CASES_ARE_VACC), 1, atol=0.00001)

    # What proportion of cases within vaccination or unvaccinated status are for each age?
    age_df["P_of_Cases(Vacc)"] = age_df["P_of_All_Cases(Vacc)"] / age_df["P_of_All_Cases(Vacc)"].sum()
    age_df["P_of_Cases(NonVacc)"] = age_df["P_of_All_Cases(NonVacc)"] / age_df["P_of_All_Cases(NonVacc)"].sum()
    assert all(age_df["P_of_Cases(Vacc)"] > 0)
    assert all(age_df["P_of_Cases(NonVacc)"] > 0)

    # Calculate the probability of a vaccinated case going to the hospital by age
    age_df["P_of_Vacc_Cases_H"] = (age_df["Prop_H"] * PROPORTION_VACCINATED_HOSPITALIZED) / age_df["P_of_Cases(Vacc)"]
    age_df["P_of_NonVacc_Cases_H"] = (age_df["Prop_H"] * PROPORTION_NONVACCINATED_HOSPITALIZED) / age_df[
        "P_of_Cases(NonVacc)"
    ]

    # Assertions: Amount hospitalized must equal the expected value
    target = age_df["P_of_All_Cases(Vacc)"].sum() * PROPORTION_VACCINATED_HOSPITALIZED
    value = (age_df["P_of_Vacc_Cases_H"] * age_df["P_of_All_Cases(Vacc)"]).sum()
    assert np.isclose(target, value, atol=0.0000001)
    target = age_df["P_of_All_Cases(NonVacc)"].sum() * PROPORTION_NONVACCINATED_HOSPITALIZED
    value = (age_df["P_of_NonVacc_Cases_H"] * age_df["P_of_All_Cases(NonVacc)"]).sum()
    assert np.isclose(target, value, atol=0.0000001)

    # STEP 2: Find the probability of a hospitalization being severe or critical
    age_df["P_Vacc_Critical"] = PROPORTION_VACC_IN_ICU * age_df["P_of_Vacc_Cases_H"]
    age_df["P_Vacc_Severe"] = (1 - PROPORTION_VACC_IN_ICU) * age_df["P_of_Vacc_Cases_H"]
    age_df["P_NonVacc_Critical"] = PROPORTION_NONVACC_IN_ICU * age_df["P_of_NonVacc_Cases_H"]
    age_df["P_NonVacc_Severe"] = (1 - PROPORTION_NONVACC_IN_ICU) * age_df["P_of_NonVacc_Cases_H"]

    # SEVERE + CRTICAL = Prob_Hospitalized
    value = age_df[["P_Vacc_Severe", "P_Vacc_Critical"]].sum(axis=1).round(5)
    target = age_df["P_of_Vacc_Cases_H"].round(5)
    assert all(value == target)
    value = age_df[["P_NonVacc_Severe", "P_NonVacc_Critical"]].sum(axis=1).round(5)
    target = age_df["P_of_NonVacc_Cases_H"].round(5)
    assert all(value == target)

    # Make sure PROPORTION_IN_ICU of all hospitalizations go to ICU
    assert np.isclose(
        (age_df["P_of_Cases(Vacc)"] * age_df["P_Vacc_Critical"]).sum(),
        vacc_params[COVIDState.CRITICAL],
        atol=0.000001,
    )
    assert np.isclose(
        (age_df["P_of_Cases(Vacc)"] * age_df["P_Vacc_Severe"]).sum(),
        vacc_params[COVIDState.SEVERE],
        atol=0.000001,
    )
    assert np.isclose(
        (age_df["P_of_Cases(NonVacc)"] * age_df["P_NonVacc_Critical"]).sum(),
        nonvacc_params[COVIDState.CRITICAL],
        atol=0.000001,
    )
    assert np.isclose(
        (age_df["P_of_Cases(NonVacc)"] * age_df["P_NonVacc_Severe"]).sum(),
        nonvacc_params[COVIDState.SEVERE],
        atol=0.000001,
    )
    return age_df


if __name__ == "__main__":
    # Reported Cases by Age
    cases_by_age = pd.Series(CovidParameters().covid_age_distribution["distribution"], index=[0, 1, 2])

    # Hospitalizations by Age
    age_df = pd.DataFrame(CovidParameters().covid_hosp_age_distribution["distribution"], index=[0, 1, 2])
    age_df.columns = ["Prop_H"]

    # Proportion of all cases that are vaccinated or unvaccinated by age: must sum to 1
    vacc_case_proportions, nonvacc_case_proportions = make_case_proportions(cases_by_age)

    # Create Vaccination Probabilities
    age_df = create_age_df(age_df, vacc_case_proportions, nonvacc_case_proportions)

    # Using the age_df, create the CDF for each combinaiton of Vacc/No-Vacc And Age Group
    params = {}
    for text in ["NonVacc", "Vacc"]:
        params[text] = {}
        for age_group in [0, 1, 2]:
            if text == "Vacc":
                params[text][age_group] = vacc_params.copy()
            else:
                params[text][age_group] = nonvacc_params.copy()
            severe = age_df.loc[age_group, f"P_{text}_Severe"]
            critical = age_df.loc[age_group, f"P_{text}_Critical"]
            params[text][age_group][COVIDState.SEVERE] = severe
            params[text][age_group][COVIDState.CRITICAL] = critical
            params[text][age_group][COVIDState.MILD] = (
                1 - severe - critical - params[text][age_group][COVIDState.ASYMPTOMATIC]
            )
            print(f"For Reported and {text}inated Parameters, age group: {age_group}: ")
            values = list(params[text][age_group].values())
            values = np.array(values).round(5).cumsum()
            values[-1] = 1
            print(list(values))

    # ------------------------------------------------------------------------------------------------------------------
    # The following is for testing purposes only.
    # ------------------------------------------------------------------------------------------------------------------

    # Testing Time. We have two questions:
    # 1. Are hospitalizations appropriately distributed by age?
    # 2. Do we match the overall hospitalization rate suggestion by the CDC data of 7.3%?
    # --- Note that only 1/4 infections are reported. This is a ~1.8% overall hospitalization rate
    example_cases = 1000
    t1 = pd.DataFrame(cases_by_age)
    t1.columns = ["Cases_By_Age"]
    t1["Cases"] = t1["Cases_By_Age"] * example_cases
    t1["Vaccinated_Cases"] = example_cases * age_df["P_of_All_Cases(Vacc)"]
    t1["Unvaccinated_Cases"] = example_cases * age_df["P_of_All_Cases(NonVacc)"]
    # Assert we still have 1000 cases
    assert np.isclose(t1[["Vaccinated_Cases", "Unvaccinated_Cases"]].sum().sum(), 1000, atol=0.0000001)

    # Question 2a: Are vaccinated cases hospitalized at the correct rate?
    t1["Vaccine_Cases_Hospitalized"] = t1["Vaccinated_Cases"] * age_df["P_of_Vacc_Cases_H"]
    value = t1["Vaccine_Cases_Hospitalized"].sum() / t1["Vaccinated_Cases"].sum()
    target = PROPORTION_VACCINATED_HOSPITALIZED
    assert np.isclose(value, target, atol=0.00001)

    # Question 2b: Are overall cases hospitalized at the correct rate?
    t1["NonVaccine_Cases_Hospitalized"] = t1["Unvaccinated_Cases"] * age_df["P_of_NonVacc_Cases_H"]
    value = (t1[["Vaccine_Cases_Hospitalized", "NonVaccine_Cases_Hospitalized"]].sum() / example_cases).sum()
    target = PROPORTION_OVERALL_HOSPITALIZED
    assert np.isclose(value, target, atol=0.00001)

    # NOTE: If Vaccinated and overall pass a test, then unvaccinated would by default pass a test

    # Question 1: Does the age distribution work out?
    hospitalizations_by_age_count = t1[["Vaccine_Cases_Hospitalized", "NonVaccine_Cases_Hospitalized"]].sum(axis=1)
    hospitalizations_by_age_rate = hospitalizations_by_age_count / hospitalizations_by_age_count.sum()
    values = hospitalizations_by_age_rate.values
    targets = age_df["Prop_H"].loc[[0, 1, 2]].values
    np.testing.assert_allclose(values, targets, atol=0.0001)
