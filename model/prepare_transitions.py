import numpy as np
import pandas as pd

import src.data_input as di
from model.state import LocationCategories


age_1_columns = ["Age Group 45 - 64 years"]
age_2_columns = ["Age Group 65 - 84 years", "Age Group 85 or more years"]
nh_beds = di.nursing_homes().Beds.sum()
lt_beds = di.ltachs().Beds.sum()
hospitals = di.hospitals().set_index("Name")
nursing_homes = di.nursing_homes()


def add_total(df):
    df["Total"] = df.sum(axis=1)
    return df


def distribute_discharges(counties, discharges: pd.DataFrame, a0_p: float, a1_p: float, a2_p: float) -> list:
    """Given the number of admissions required for a specific hospital, find the number of people from each age group
    that need to go to that hospital from each county.

    Args:
        discharges (pd.DataFrame): A discharge DataFrame from the SHEPS center
        a0_p (float): The overall probability that someone in that facility type is 0 < 50 years old
        a1_p (float): The overall probability that someone in that facility type is 50 < 65 years old
        a2_p (float): The overall probability that someone in that facility type is 65+ years old
    """
    a_list = []
    for county in counties:
        value = discharges.loc[county, "Final_Total"]
        a0 = value * a0_p
        a1 = value * a1_p
        a2 = value * a2_p
        a_list.extend([a0, a1, a2])
    return a_list


def stach_transitions(fbf, params, hospital: str, demographic_breakdown: pd.DataFrame) -> list:
    """Calulate the stach discharge probabilities for a specific hospital.

    Args:
        hospital (str): The name of the hospital (as it appears in the SHEPS data)
        breakdown (pd.DataFrame): A demographic breakdown of discharges from a hospital - provided by SHEPS
    """
    hospital_to_hospital = fbf.loc["HOSPITAL", "HOSPITAL"]
    hospital_to_nh = fbf.loc["HOSPITAL", "NH"] - fbf.loc["NH", "HOSPITAL"] * params.location.nh_st_nh
    hospital_to_lt = fbf.loc["HOSPITAL", "LT"]
    hospital_movement = fbf.loc["HOSPITAL"].sum() - fbf.loc["NH", "HOSPITAL"] * params.location.nh_st_nh

    temp_total = demographic_breakdown[hospital].max()
    a1 = demographic_breakdown.loc[age_1_columns, hospital].sum() * 0.75 / temp_total
    a2 = demographic_breakdown.loc[age_2_columns, hospital].sum() / temp_total

    rows = []
    for age_group in params.age_groups:
        # Whats the probability of being discharged to a hospital?
        to_hospital = hospital_to_hospital / hospital_movement
        if age_group == 2:
            # LT patients are primarily 65p: we also need more 65p so we can have enough LT to NH movement
            to_lt = (params.location.lt_65p / a2) * (hospital_to_lt / hospital_movement)
            # All NH movement must be 65+
            to_nh = (1 / a2) * (hospital_to_nh / hospital_movement)
        elif age_group == 1:
            to_lt = ((1 - params.location.lt_65p) / a1) * (hospital_to_lt / hospital_movement)
            to_nh = 0
        else:
            to_nh, to_lt = 0, 0
        row = [0, to_hospital, to_lt, to_nh]
        row[0] = 1 - sum(row)
        row = [hospital, age_group] + row
        rows.append(row)
    return rows


def merge_hospital_and_demographic_files():
    """Merge the demographic information with the hospitals file"""
    breakdown = di.demographics().set_index("Category").T
    columns = ["Patient Residence State NC", "Patient Residence State Not NC"]
    breakdown["Total_Patients"] = breakdown[columns].sum(axis=1)
    breakdown["NC_Proportion"] = breakdown["Patient Residence State NC"] / breakdown["Total_Patients"]
    # Merge 1
    hospital_df = pd.merge(hospitals, breakdown[["Total_Patients", "NC_Proportion"]], left_index=True, right_index=True)
    # Read the discharge file
    hospital_los = di.total_discharges().T
    hospital_los.columns = ["Newborn Total", "Total", "Newborn LOS", "LOS"]
    # Merge 2
    return pd.merge(hospital_df, hospital_los["LOS"], left_index=True, right_index=True)


def make_hospital_df(params):
    hospital_df = merge_hospital_and_demographic_files()

    # ----- Step 1: Subset to only important columns
    columns = ["Beds", "ICU Beds", "Total_Patients", "NC_Proportion", "LOS", "Acute Fill", "ICU Fill"]
    hospital_df = hospital_df[columns].copy()
    hospital_df = hospital_df.rename(columns={"ICU Beds": "ICU_Beds"})
    hospital_df["Acute_Beds"] = hospital_df["Beds"] - hospital_df["ICU_Beds"]
    assert hospital_df["Acute_Beds"].min() >= 0

    # ----- Step 2: Fill Acute Beds and Match Input Parameter
    # Acute agents are based on the acute_fill_proportion parameter
    if params.location.use_real_data:
        hospital_df["Acute_Agents"] = round(hospital_df.Acute_Beds * hospital_df["Acute Fill"]).astype(int)
    else:
        hospital_df["Acute_Agents"] = (
            hospital_df.Acute_Beds / hospital_df.Beds * hospital_df.Total_Patients * hospital_df.LOS / 365
        )
        ratio = params.location.acute_fill_proportion / (hospital_df.Acute_Agents.sum() / hospital_df.Acute_Beds.sum())
        hospital_df["Acute_Agents"] *= ratio
        hospital_df["Acute_Agents"] = round(hospital_df["Acute_Agents"]).astype(int)
    # Some hospitals may have too many acute patients. Fix any hospitals with more agents than beds
    temp = hospital_df[hospital_df["Acute_Agents"] > hospital_df["Acute_Beds"]].copy()
    temp["Acute_Agents"] = temp["Acute_Beds"]
    hospital_df.loc[temp.index] = temp
    assert all(hospital_df["Acute_Agents"] <= hospital_df["Acute_Beds"])
    # Split This Between NC and Non-NC Agents
    hospital_df["Acute_NC_Agents"] = np.ceil(hospital_df.Acute_Agents * hospital_df.NC_Proportion).astype(int)
    hospital_df["Acute_Non_NC_Agents"] = hospital_df.Acute_Agents - hospital_df.Acute_NC_Agents
    assert hospital_df.Acute_Agents.sum() == hospital_df[["Acute_NC_Agents", "Acute_Non_NC_Agents"]].sum().sum()

    # ----- Step 3: Fill ICU Beds and Match Input Parameter
    if params.location.use_real_data:
        hospital_df["ICU_Agents"] = round(hospital_df.Acute_Beds * hospital_df["ICU Fill"]).astype(int)
    else:
        hospital_df["ICU_Agents"] = (
            (hospital_df.ICU_Beds / hospital_df.Beds) * hospital_df.Total_Patients * hospital_df.LOS / 365
        )
        ratio = params.location.icu_fill_proportion / (hospital_df.ICU_Agents.sum() / hospital_df.ICU_Beds.sum())
        hospital_df["ICU_Agents"] *= ratio
        hospital_df["ICU_Agents"] = round(hospital_df["ICU_Agents"])
    # Fix any hospitals with more agents than beds
    temp = hospital_df[hospital_df["ICU_Agents"] > hospital_df["ICU_Beds"]].copy()
    temp["ICU_Agents"] = temp["ICU_Beds"]
    hospital_df.loc[temp.index] = temp
    assert all(hospital_df["ICU_Agents"] <= hospital_df["ICU_Beds"])
    # Split This Between NC and Non-NC Agents
    hospital_df["ICU_NC_Agents"] = np.ceil(hospital_df.ICU_Agents * hospital_df.NC_Proportion).astype(int)
    hospital_df["ICU_Non_NC_Agents"] = hospital_df.ICU_Agents - hospital_df.ICU_NC_Agents
    assert hospital_df.ICU_Agents.sum() == hospital_df[["ICU_NC_Agents", "ICU_Non_NC_Agents"]].sum().sum()

    total_agents = hospital_df[["Acute_NC_Agents", "ICU_NC_Agents"]].sum(axis=1)
    hospital_df["NC_Yearly_Adm"] = total_agents.div(hospital_df["LOS"] + 0.25) * 365
    return hospital_df


def make_fbf(params, demographic_breakdown, discharge_total, hospital_df):
    """Here we make a matrix of transitions between the 4 categories: Community, HOSPITAL, LT, and NH
    We are calculating the number of people that transition from 1 category to the next
    """

    # The following list map SHEPs rows to specific categories
    stach_columns = [
        "Patient Disposition Discharged, transferred to acute facility",
        "Patient Disposition Discharged, transferred",
    ]
    # lt_columns = ["Patient Disposition Discharged, transferred to long term acute care facility (LTAC)"]
    nh_columns = [
        "Patient Disposition Discharged, transferred to facility that provides nursing, custodial, or supportive care"
    ]
    death_columns = ["Patient Disposition Expired"]

    # Calculate the proportion of people going from hospital to category X
    hospital_stach_proportion = demographic_breakdown.loc[stach_columns, "Total"].sum() / discharge_total
    hospital_nh_proportion = demographic_breakdown.loc[nh_columns, "Total"].sum() / discharge_total
    hospital_dead_proportion = demographic_breakdown.loc[death_columns, "Total"].sum() / discharge_total
    # Calculate the number of transfers
    admissions = hospital_df.NC_Yearly_Adm.sum()
    hospital_to_hospital = admissions * hospital_stach_proportion
    hospital_to_nh = admissions * hospital_nh_proportion
    hospital_to_death = admissions * hospital_dead_proportion
    # We need enough LTACH to keep the facilities full
    hospital_to_ltach = params.location.lt_fill_proportion * lt_beds * 365 / params.location.LT_LOS["mean"]
    hospital_to_com = admissions - hospital_to_hospital - hospital_to_ltach - hospital_to_nh - hospital_to_death

    # LTACH
    # Assumption: LTACH admissions = SUM(STACH TO LTACH discharges)
    lt_admissions = hospital_to_ltach
    lt_to_hospital = lt_admissions * params.location.lt_to_hospital
    lt_to_ltach = 0
    lt_to_nh = lt_admissions * params.location.lt_to_nh
    lt_to_death = lt_admissions * params.location.lt_death
    lt_to_com = lt_admissions - lt_to_hospital - lt_to_nh - lt_to_death
    # NH
    # Assumption: Equal admissions and discharges for NHs
    # #TODO We can't use the community_to_nh param right now - it's way to high: params.location.community_to_nh
    # Calculate a realistic number instead
    nh_avg_los = pd.Series(di.nh_los()).mean()
    nh_admissions = nursing_homes["average_number_of_residents_per_day"].sum() * 365 / nh_avg_los
    community_to_nh_temp = nh_admissions - hospital_to_nh - lt_to_nh
    nh_to_com = nh_admissions * ((1 - params.location.nh_death_proportion) * params.location.nh_to_community)
    nh_to_death = nh_admissions * params.location.nh_death_proportion
    nh_to_hospital = nh_admissions - nh_to_com - nh_to_death
    nh_to_ltach = 0  # NH to LT is not allowed: Team Assumption
    nh_to_nh = 0  # NH to NH is not allowed: Team Assumption
    # COMMUNITY
    com_to_com = 0
    com_to_hospital = admissions - hospital_to_hospital - lt_to_hospital - nh_to_hospital
    com_to_ltach = 0
    com_to_death = 0
    # Make 4x4
    r1 = [com_to_com, com_to_hospital, com_to_ltach, community_to_nh_temp, com_to_death]
    r2 = [hospital_to_com, hospital_to_hospital, hospital_to_ltach, hospital_to_nh, hospital_to_death]
    r3 = [lt_to_com, lt_to_hospital, lt_to_ltach, lt_to_nh, lt_to_death]
    r4 = [nh_to_com, nh_to_hospital, nh_to_ltach, nh_to_nh, nh_to_death]
    fbf_columns = ["COMMUNITY", "HOSPITAL", "LT", "NH", "Death"]
    fbf = pd.DataFrame([r1, r2, r3, r4], columns=fbf_columns)
    fbf.index = fbf_columns[:-1]

    # TESTS: The 4-by-4 must be aligned. Equal numbers in and out for all 4 locations
    assert np.isclose(
        fbf[["COMMUNITY", "Death"]].sum().sum(), fbf.loc["COMMUNITY"].sum(), rtol=0.01
    ), "4-by-4 is not aligned for community."
    for i in ["HOSPITAL", "LT", "NH"]:
        assert np.isclose(fbf[[i]].sum().sum(), fbf.loc[i].sum(), rtol=0.01), f"4-by-4 is not aligned for {i}"
    return fbf


def prepare_transitions(params):
    counties = params.counties
    age_groups = params.age_groups

    # ----- Synthetic Population
    syn_pop = di.read_population()
    syn_pop_counts = syn_pop.groupby(["County_Code", "Age_Group"]).size()
    syn_pop_counts.columns = ["Population"]

    # ---------------
    # The Hospital DF
    # ---------------
    hospital_df = make_hospital_df(params)

    # ---- Read SHEPs
    # By Demographic
    demographic_breakdown = di.demographics().set_index("Category")
    demographic_breakdown = add_total(demographic_breakdown[list(hospitals.index)].copy())
    # By County
    discharges = di.county_discharges()
    discharges = add_total(discharges[list(hospitals.index)].copy())

    # Calculate Discharges
    temp_columns = ["Patient Residence State NC", "Patient Residence State Not NC"]
    discharge_total = demographic_breakdown.loc[temp_columns]["Total"].sum()
    # Calculate the percent of people from each age group
    p_50_64 = demographic_breakdown.loc[age_1_columns, "Total"].sum() * 0.75 / discharge_total  # .75 b/c age_1 is 45-64
    p_G65 = demographic_breakdown.loc[age_2_columns, "Total"].sum() / discharge_total
    p_L50 = 1 - p_50_64 - p_G65

    # -----------------
    # THE Four BY Four
    # -----------------
    fbf = make_fbf(params, demographic_breakdown, discharge_total, hospital_df)

    # ---------------------
    # Community Transitions
    # ---------------------
    # Goal - Calculate the probability of transition from the community to a facility for each day (by age/county)

    # Step 1: Adjust the totals so that there is enough Community -> Hospital
    for _, row in hospital_df.iterrows():
        discharge_total = discharges[[_]].sum()
        needed_total = hospital_df.loc[_].NC_Yearly_Adm
        ratio = (needed_total / discharge_total).values[0]
        discharges[[_]] = (round(discharges[[_]] * ratio)).astype(int)

    discharges["Adjusted_Total"] = discharges[[i for i in hospital_df.index]].sum(axis=1)
    discharges["Final_Total"] = (discharges["Adjusted_Total"] / discharges["Adjusted_Total"].sum()) * fbf.loc[
        "COMMUNITY"
    ]["HOSPITAL"]

    # Community: Only Community -> Hospital and Community -> NH is possible
    community = pd.DataFrame(
        [[county, age_group] for county in counties for age_group in age_groups], columns=["County_Code", "Age_Group"]
    ).set_index(["County_Code", "Age_Group"])
    community["Population"] = syn_pop_counts
    temp = distribute_discharges(counties, discharges, p_L50, p_50_64, p_G65)
    community["HOSPITAL"] = temp / community["Population"] / 365
    # How many are 65+, and take into account several of these people are not in the community
    g65 = community[community.index.get_level_values(1) == 2].Population.sum()
    g65 -= nh_beds * (nursing_homes.average_number_of_residents_per_day.sum() / nursing_homes.Beds.sum())
    community["NH"] = [0, 0, fbf.loc[LocationCategories.COMMUNITY.name][LocationCategories.NH.name] / g65 / 365] * 100

    # --------------------
    # Facility Transitions
    # --------------------

    # ----- Hospitals
    all_lists = []
    for hospital in [item for item in hospitals.index]:
        all_lists.extend(stach_transitions(fbf, params, hospital, demographic_breakdown))

    # ----- NH
    nh_movement = fbf.loc["NH"]["COMMUNITY"] + fbf.loc["NH"]["HOSPITAL"]
    nh_to_com_proportion = fbf.loc["NH"]["COMMUNITY"] / nh_movement
    nh_to_stach_proportion = 1 - nh_to_com_proportion
    for age_group in age_groups:
        # NH must be 65+
        if age_group < 2:
            row = ["NH", age_group, 1, 0, 0, 0]
        else:
            row = ["NH", age_group, nh_to_com_proportion, nh_to_stach_proportion, 0, 0]
        all_lists.append(row)

    # ----- LT
    lt_movement = fbf.loc["LT"][["COMMUNITY", "HOSPITAL", "NH"]].sum()
    lt_to_hospital_proportion = fbf.loc["LT"]["HOSPITAL"] / lt_movement
    for age_group in age_groups:
        # We don't allow <50 to go to LTACH.
        if age_group == 0:
            row = ["LT", age_group, 1, 0, 0, 0]
        else:
            # All lt_to_nh_p must be contained by 65+, <65 gets a 0.
            if age_group == 2:
                lt_to_nh_proportion = (1 / params.location.lt_65p) * (fbf.loc["LT"]["NH"] / lt_movement)
            else:
                lt_to_nh_proportion = 0
            # All
            lt_to_community_proportion = 1 - lt_to_nh_proportion - lt_to_hospital_proportion
            row = ["LT", age_group, lt_to_community_proportion, lt_to_hospital_proportion, 0, lt_to_nh_proportion]
        all_lists.append(row)

    # Tests: All rows must add to 1.
    for row in all_lists:
        assert np.isclose(sum(row[2:]), 1, rtol=0.001), f"Error w/ {row}!. It sums to {round(sum(row[2:]), 5)}, not 1"

    facility_transitions = pd.DataFrame(
        all_lists, columns=["Facility", "Age_Group", "COMMUNITY", "HOSPITAL", "LT", "NH"]
    )

    # LTACH TESTING
    lt_df = facility_transitions[(facility_transitions.Facility == "LT") & (facility_transitions.Age_Group != 0)]
    gb = lt_df.groupby("Age_Group")["COMMUNITY"].mean()
    # The LTACH to Community Probability must average X%
    p65 = params.location.lt_65p
    lt_to_community_proportion = fbf.loc["LT"]["COMMUNITY"] / lt_movement
    assert np.isclose(gb.loc[1] * (1 - p65) + gb.loc[2] * p65, lt_to_community_proportion, rtol=0.01)
    # The LTACH to TACH probabilities must sum to X%
    assert np.isclose(lt_df[["HOSPITAL"]].sum(axis=1).mean(), params.location.lt_to_hospital, rtol=0.1)
    # TO NH is not allowed for Age != 2
    assert np.isclose(
        facility_transitions[facility_transitions.Facility == "LT"].groupby("Age_Group").sum().loc[[0, 1]].NH.sum(),
        0,
        rtol=0,
    )
    # NH TESTING
    nh_df = facility_transitions[(facility_transitions.Facility == "NH") & (facility_transitions.Age_Group == 2)]
    assert nh_df.NH.sum() == 0  # NH to NH is impossible
    assert nh_df.LT.sum() == 0  # NH to LT is impossible
    assert nh_df.COMMUNITY.value_counts().shape[0] == 1  # All community values are equal

    # ----------------
    # Death Dictionary
    # ----------------
    death_dictionary = {}
    for item in fbf.iterrows():
        death_dictionary[item[0]] = item[1].Death / item[1].sum()

    return {
        "hospital_df": hospital_df,
        "discharges_df": discharges.drop(["Total", "Adjusted_Total", "Final_Total"], axis=1),
        "community_transitions": community,
        "facility_transitions": facility_transitions,
        "death_dictionary": death_dictionary,
        "four_by_four": fbf,
        "hospital_age_distribution": [p_L50, p_L50 + p_50_64, 1],
    }
