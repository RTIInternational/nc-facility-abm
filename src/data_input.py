import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from model.state import LocationCategories

from src.misc_functions import get_inverted_distance_probabilities

filepaths = yaml.load(Path("config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)


@lru_cache()
def read_population():
    pop = pd.read_parquet(filepaths["synthetic_population_file_parquet"]["path"])
    return pop


@lru_cache()
def sample_population(variables: tuple, limit: int, seed: int):
    variables = list(set([i for i in variables] + ["County_Code"]))
    population_data = read_population()[variables]
    if limit < len(population_data):
        return population_data.sample(limit, random_state=seed).reset_index(drop=True)
    return population_data


def combine_hospitals(df) -> pd.DataFrame:
    """Several Hospitals from 2018 need to be combined for 2021
    This occurs because the NCDHSR file has hospitals grouped by license number
    The older SHEPs file, may have hospitals disaggregated
    """

    df["Johnston Health Clayton"] += df["Johnston Health Smithfield"]
    df = df.drop(["Johnston Health Smithfield"], axis=1)
    df["University of North Carolina Hospitals"] += df["UNC Hillsborough"]
    df = df.drop(["UNC Hillsborough"], axis=1)
    df["WakeMed"] += df["WakeMed North Family Health& Woman's Hospital"]
    df = df.drop(["WakeMed North Family Health& Woman's Hospital"], axis=1)
    # https://www.novanthealth.org/kernersville-medical-center.aspx
    df["Novant Health Forsyth Medical Center"] += df["Novant Health Kernersville Medical Center"]
    df = df.drop(["Novant Health Kernersville Medical Center"], axis=1)
    # https://www.novanthealth.org/clemmons-medical-center.aspx
    df["Novant Health Forsyth Medical Center"] += df["Novant Health Clemmons Medical Center"]
    df = df.drop(["Novant Health Clemmons Medical Center"], axis=1)
    # https://www.novanthealth.org/charlotte-orthopedic-hospital.aspx
    df["Novant Health Presbyterian Medical Center"] += df["Novant Health Charlotte Orthopedic Hospital"]
    df = df.drop(["Novant Health Charlotte Orthopedic Hospital"], axis=1)
    df["First Health Moore Regional Hospital"] += df["FirstHealth Moore Regional Hospital - Hamlet"]
    df = df.drop(["FirstHealth Moore Regional Hospital - Hamlet"], axis=1)
    # https://www.bizjournals.com/charlotte/news/2019/07/05/atrium-health-combining-two-hospitals-in-region.html
    df["Cleveland Regional Medical Center"] += df["Carolinas HealthCare System Kings Mountain Hospital"]
    df = df.drop(["Carolinas HealthCare System Kings Mountain Hospital"], axis=1)
    # https://myharnetthealth.org/about-hhs/
    df["Betsy Johnson Hospital"] += df["Central Harnett Health"]
    df = df.drop(["Central Harnett Health"], axis=1)
    # This is an LTACH
    df = df.drop(["ContinueCare University NC"], axis=1)
    return df


def county_discharges() -> pd.DataFrame:
    nc = nc_counties()
    df = pd.read_csv(filepaths["county_discharges"]["path"]).fillna(0)
    exc = ["VIRGINIA", "TENNESSEE", "Other/Missing", "SOUTH CAROLINA", "Actual", "Calculated", "Unreported", "GEORGIA"]
    df = df[[i not in exc for i in df.RESIDENCE.values]]
    df = df.merge(nc[["County_Code", "County"]], left_on="RESIDENCE", right_on="County")
    df = df.set_index("County_Code").drop(["County", "RESIDENCE"], axis=1)
    return combine_hospitals(df)


def total_discharges() -> pd.DataFrame:
    # DO NOT COMBINE. This has a LOS row. If you combine, the LOS values are added together.
    df = pd.read_csv(filepaths["total_discharges"]["path"])
    df = df.drop(["Summary Data for All Hospitals"], axis=1)
    return df[county_discharges().columns]


def demographics() -> pd.DataFrame:
    df = pd.read_csv(filepaths["sheps_demographics"]["path"])
    df = df.drop(["Actual", "Calculated", "Difference"], axis=1)
    return combine_hospitals(df)


def county_hospital_distances() -> Dict[int, Any]:
    fp = Path(filepaths["geography_folder"]["path"], "county_HOSPITAL_distances_sorted.json")
    county_to_hospital_distances = json.loads((fp).read_text())
    county_to_hospital_distances = {int(k): v for k, v in county_to_hospital_distances.items()}
    return county_to_hospital_distances


def county_facility_distribution(loc_type: LocationCategories, closest_n=5) -> Dict[int, Any]:
    fp = Path(filepaths["geography_folder"]["path"], f"county_{loc_type}_distances_sorted.json")
    # read in the distances
    county_to_facility_distances = json.loads((fp).read_text())

    if loc_type == LocationCategories.LT.name:
        bed_dict = ltachs().set_index("Name")["Beds"].to_dict()
    elif loc_type == LocationCategories.NH.name:
        bed_dict = nursing_homes().set_index("Name")["Beds"].to_dict()
    elif loc_type == LocationCategories.HOSPITAL.name:
        hos_df = hospitals()
        bed_dict = hos_df.set_index("Name")["Beds"].to_dict()
    else:
        print(loc_type)
        raise ValueError("ERROR: loc type invalid")

    probability_distributions = {}
    # for each county, get a dataframe of LTACH, distance, bedcount
    for county, county_list in county_to_facility_distances.items():
        county_list = [i for i in county_list if i["Name"] in bed_dict.keys()]
        county_dict = [
            {"Name": val["Name"], "beds": bed_dict[val["Name"]], "distance": val["distance_mi"]}
            for val in county_list[0:closest_n]
        ]
        names = [i["Name"] for i in county_dict]
        distances = [i["distance"] for i in county_dict]
        beds = [i["beds"] for i in county_dict]
        distance_weight = [1 / (i / min(distances)) for i in distances]
        bed_weight = [i / max(beds) for i in beds]
        weight = [a * b for a, b in zip(distance_weight, bed_weight)]
        prob_array = [i / sum(weight) for i in weight]

        probability_distributions[int(county)] = {
            "names": names,
            "prob_array": prob_array,
            "cdf": list(np.cumsum(prob_array)),
        }

    return probability_distributions


def facility_to_county_probabilities(loc_type) -> Dict[int, Any]:
    # get the distance between each LTACH and each county
    fp = Path(filepaths["geography_folder"]["path"], f"county_{loc_type}_distances_sorted.json")
    county_to_facility_distances = json.loads((fp).read_text())
    # invert it to get the distances between NH and all counties
    return get_inverted_distance_probabilities(county_to_facility_distances)


def nh_los() -> pd.DataFrame:
    nh_los = pd.read_csv(filepaths["nh_los"]["path"])
    nh_los = nh_los[(0 < nh_los.los) & (nh_los.los < 2000)].copy()
    nh_los = nh_los.astype(int)
    nh_los.cfreq = nh_los.cfreq.apply(lambda x: int(x * 0.1))
    a_list = []
    for row in nh_los.itertuples():
        a_list.extend([row.los] * int(row.cfreq))
    return a_list


def hospitals(drop_na=True) -> pd.DataFrame:
    df = pd.read_csv(filepaths["hospitals"]["path"])
    if drop_na:
        df = df[~df["ptorg name"].isna()]
    df["Beds"] = df["Beds"].astype(int)
    df["ICU Beds"] = df["ICU Beds"].astype(int)
    df.rename(columns={"ptorg name": "Name"}, inplace=True)
    return df


def nursing_homes() -> pd.DataFrame:
    df = pd.read_csv(filepaths["nursing_homes"]["path"])
    # Nursing homes with missing resident informaiton are assumed to be 50% full
    temp_df = df[df["average_number_of_residents_per_day"].isna()].copy()
    temp_df["average_number_of_residents_per_day"] = temp_df["Beds"] / 2
    df.loc[temp_df.index] = temp_df
    return df


def ltachs() -> pd.DataFrame:
    df = pd.read_csv(filepaths["ltachs"]["path"])
    return df


def nc_counties() -> pd.DataFrame:
    return pd.read_csv(filepaths["nc_counties"]["path"])
