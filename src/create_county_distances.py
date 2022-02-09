from os import read
from pathlib import Path

import pandas as pd
import yaml

filepaths = yaml.load(Path("config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)


def read_county_distance_data() -> pd.DataFrame:
    """
    Reads dataset containing distances from county to county in the US. Filters to NC counties only and replaces
    FIPS codes with shortened county codes.
    """
    county_codes = pd.read_csv(filepaths["nc_counties"]["path"])
    county_codes["FIPS"] = county_codes.County_Code + 37000
    county_codes = county_codes[["County", "County_Code", "FIPS"]]

    county_distances = pd.read_csv(filepaths["county_distances_base"]["path"])

    nc_distances = county_distances.merge(county_codes, how="inner", left_on="county1", right_on="FIPS").drop(
        "FIPS", axis=1
    )
    nc_distances.rename(columns={"County_Code": "county_from", "mi_to_county": "distance"}, inplace=True)
    nc_distances = nc_distances.merge(county_codes, how="inner", left_on="county2", right_on="FIPS").drop(
        "FIPS", axis=1
    )
    nc_distances.rename(columns={"County_Code": "county_to"}, inplace=True)
    nc_distances = nc_distances[["county_from", "county_to", "distance"]]

    outpath = Path(filepaths["county_distances"]["path"])
    nc_distances.to_csv(outpath, index=False)


if __name__ == "__main__":
    read_county_distance_data()
