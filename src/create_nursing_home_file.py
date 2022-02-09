import urllib.parse
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import requests
import yaml

filepaths = yaml.load(Path("config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)


def call_cms_api(state: str, offset: int, url: str, database_id: str) -> pd.DataFrame:

    if len(state) > 2:
        raise ValueError(f"State string '{state}' is too long. Use the two-letter state abbreviation, e.g. 'NC'.")

    headers = {"accept": "application/json"}
    params = (
        (
            "query",
            f'[SELECT * FROM {database_id}][WHERE provider_state == "{state}"][LIMIT 500 OFFSET {offset}];',
        ),
        ("show_db_columns", "true"),
    )
    response = requests.get(url=url, headers=headers, params=params)
    json = response.json()

    return pd.DataFrame.from_dict(json)


def merge_county_codes(df: pd.DataFrame, left_on: str):
    county_codes = pd.read_csv(filepaths["nc_counties"]["path"])
    county_codes["FIPS"] = county_codes.County_Code + 37000
    county_codes["FIPS"] = county_codes["FIPS"].astype(int)
    county_codes["County_Code"] = county_codes["County_Code"].astype(int)
    county_codes = county_codes[["County", "County_Code", "FIPS"]]

    return df.merge(county_codes, how="left", left_on=left_on, right_on="County")


def convert_str_col_to_float(series: pd.Series) -> pd.Series:
    """
    Many of the numeric columns in the CMS data are read as strings from the API.
    Converts such columns to floats.
    """
    series = np.where(series.eq(""), None, series)
    series = series.astype(float)
    return series


def add_staffing_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns showing total staffing hours per week for each nursing home by staff
    category. Calculates these using CMS data on average number of residents per day
    and reported staffing hours per resident per day.
    """
    staff_types = [
        "nurse_aide",
        "lpn",
        "rn",
        "physical_therapist",
    ]
    df["average_number_of_residents_per_day"] = convert_str_col_to_float(df["average_number_of_residents_per_day"])
    for staff_type in staff_types:
        df[f"reported_{staff_type}_staffing_hours_per_resident_per_day"] = convert_str_col_to_float(
            df[f"reported_{staff_type}_staffing_hours_per_resident_per_day"]
        )
        df[f"{staff_type}_hours_per_week"] = (
            df[f"reported_{staff_type}_staffing_hours_per_resident_per_day"]
            * df["average_number_of_residents_per_day"]
            * 7
        )
        # median impute missing values
        df[f"{staff_type}_hours_per_week"].fillna(df[f"{staff_type}_hours_per_week"].median(), inplace=True)
        df[f"{staff_type}_hours_per_week"] = df[f"{staff_type}_hours_per_week"].round().astype(int)
    df["total_staff_hours_per_week"] = sum([df[f"{staff_type}_hours_per_week"] for staff_type in staff_types])
    return df


def make_cms_file(
    state: str = "NC",
    url: str = "https://data.cms.gov/provider-data/api/1/datastore/sql",
    database_id: str = "8c6cdead-ca61-505e-b81e-177452840a72",
):
    state = state.upper()
    dfs = []
    offset = 0
    dfs.append(call_cms_api(state=state, offset=offset, url=url, database_id=database_id))

    # The CMS API limits queries to 500 rows. Here we continue to call the CMS API as long as the most recent query
    # returned 500 rows, presuming that once we get a query of length < 500, we've gotten all the rows.
    # This logic fails if a state happens to have a number of rows that is a multiple of 500. However, the end
    # result would not be fatal - this would simply append an empty df to the list.
    while len(dfs[-1]) == 500:
        offset += 500
        dfs.append(call_cms_api(state=state, offset=offset, url=url, database_id=database_id))

    df = pd.concat(dfs)

    # Fix weird spelling of McDowell county
    df.provider_county_name = np.where(df.provider_county_name.eq("Mc Dowell"), "McDowell", df.provider_county_name)

    # Remove hospital
    df = df[df.provider_name != "TRANSYLVANIA REGIONAL HOSPITAL"].reset_index(drop=True)

    # Add county code and staffing columns
    df = merge_county_codes(df, left_on="provider_county_name")
    df = add_staffing_cols(df)

    outpath = Path(filepaths["cms_data_folder"]["path"]).joinpath(f"cms_data_{state}.csv")
    df.to_csv(outpath, index=False)


def make_nh_file():
    """Check the CMS Data. If any new facilities exist, update the NH file and GEOCODE that location."""

    # ----- Latest CMS Data
    cms = pd.read_csv(Path(filepaths["cms_data"]["path"]))
    nh_cols = [
        "record_number",
        "federal_provider_number",
        "provider_name",
        "number_of_certified_beds",
        "average_number_of_residents_per_day",
        "provider_address",
        "provider_city",
        "provider_zip_code",
    ]
    county_code_cols = ["federal_provider_number", "County", "County_Code", "FIPS"]
    df = cms[nh_cols].copy()
    df_codes = cms[county_code_cols].copy()

    # ----- Base NH File
    nh = pd.read_csv(Path(filepaths["nursing_homes_base"]["path"]))

    # Step 1: Remove any NHs not in latest CMS data
    nh = nh[nh.federal_provider_number.isin(df.federal_provider_number)]

    # Step 2: Add any new CMS records
    df_temp = df[~df.federal_provider_number.isin(nh.federal_provider_number)].copy()

    if df_temp.shape[0] > 0:
        lat = []
        lon = []
        for _, row in df_temp.iterrows():
            address = f"{row.provider_address}, {row.provider_city}, NC, {row.provider_zip_code}"
            url = "https://nominatim.openstreetmap.org/search/" + urllib.parse.quote(address) + "?format=json"
            response = requests.get(url).json()
            if len(response) == 0:
                lat.append(np.nan)
                lon.append(np.nan)
            else:
                lat.append(response[0]["lat"])
                lon.append(response[0]["lon"])
        df_temp["lat"] = lat
        df_temp["lon"] = lon
        nh = pd.concat([nh, df_temp]).reset_index(drop=True)
        if np.nan in lat:
            print("WARNING: At least one NH was not geocoded correctly. Please visit the NH file and review.")

    # Step 3: Add county data to NHs
    nh = nh.merge(df_codes, how="left", on="federal_provider_number")

    # Step 4: Remove Hospital
    nh = nh[nh.provider_name != "Transylvania Regional Hospital"].reset_index(drop=True)

    # Step 5: Save
    nh.rename(columns={"provider_name": "Name", "number_of_certified_beds": "Beds"}, inplace=True)
    nh.Name = nh["Name"].str.title()
    nh.to_csv(filepaths["nursing_homes"]["path"], index=False)


if __name__ == "__main__":
    fire.Fire(make_cms_file)

    make_nh_file()
