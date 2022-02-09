from pathlib import Path

import numpy as np
import pandas as pd
from src.data_input import filepaths
from src.jit_functions import assign_conditions
import src.data_input as di


def main():
    """Extract the necessary columns from the 2017 synthetic population to use for the models

    This requires that you have access to the 2017 synthetic populaiton. This is not available on the repository.
    """

    # Read the 2017 Synthetic Persons and Households files
    df = pd.read_csv(
        "data/synthetic_population/37/NC2017_Persons.csv",
        usecols=["hh_id", "agep", "sex", "rac1p"],
    )

    df_household = pd.read_csv(
        "data/synthetic_population/37/NC2017_Households.csv",
        usecols=["hh_id", "logrecno", "county", "tract", "blkgrp"],
    )
    df = df.merge(df_household)

    df = df.rename(columns={"agep": "Age", "sex": "Sex", "rac1p": "Race", "county": "County_Code"})

    # Correct the Age
    df["Age_Years"] = df["Age"]
    df["Age_Group"] = -1
    df.loc[df["Age_Years"] < 50, "Age_Group"] = 0
    df.loc[df["Age_Years"] > 64, "Age_Group"] = 2
    df.loc[df["Age_Group"] == -1, "Age_Group"] = 1
    # Correct the Race
    df.loc[df["Race"] > 2, "Race"] = 3

    # Reduce Columns
    df = df[
        [
            "County_Code",
            "Sex",
            "Age_Group",
            "Race",
            "Age_Years",
            "tract",
            "blkgrp",
            "logrecno",
        ]
    ]

    # Pad the population to 10m people. The approximate population of NC.
    nc_counties = di.nc_counties()
    number_to_add = nc_counties.Population.sum() - df.shape[0]
    new_people = df.sample(number_to_add)
    df = df.append(new_people)
    df = df.reset_index(drop=True)
    df["County_Code"] = df["County_Code"].astype(int)

    # Assign Comorbidities
    rng = np.random.RandomState(1111)
    df["Comorbidities"] = assign_conditions(df.Age_Group.values, rng.rand(df.shape[0]))

    # Save as parquet file
    df.to_parquet(Path(filepaths["synthetic_population_file_parquet"]["path"]))


if __name__ == "__main__":
    main()
