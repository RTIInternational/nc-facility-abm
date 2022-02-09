from pathlib import Path

import yaml
import pandas as pd

import src.data_input as di
import submodels.covid19.src.covid_data_input as cdi

filepaths = yaml.load(Path("submodels/covid19/config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)


if __name__ == "__main__":
    # ----- Create a dataframe of adult people in NC by county and by age
    nc = di.nc_counties()
    syn_pop = di.read_population()
    syn_pop_counts = syn_pop.groupby(["County_Code", "Age_Group"]).size()
    syn_pop_counts = syn_pop_counts.reset_index()
    syn_pop_counts = syn_pop_counts.merge(nc[["County_Code", "County"]], left_on="County_Code", right_on="County_Code")
    syn_pop_counts = syn_pop_counts.drop("County_Code", axis=1)
    syn_pop_counts = syn_pop_counts.pivot(index="County", columns="Age_Group", values=0)

    # ----- Create a dataframe of vaccination counts by county and by age
    df = cdi.vaccinations_by_age()
    df = df.fillna(0)
    df["Month"] = [i[0] for i in df["Week of"].str.split("/")]
    df["Year"] = [i[2] for i in df["Week of"].str.split("/")]
    # Remove 01-2022
    df = df[~((df.Month == "1") & (df.Year == "2022"))]
    # Remove out of state and missing
    df = df[~df["County "].isin(["Missing", "Out of State"])]

    age_df = pd.DataFrame(df.groupby("County ")[["5-11", "12-17", "18-24", "25-49"]].sum().sum(axis=1))
    age_df["a"] = df.groupby("County ")["50-64"].sum()
    age_df["b"] = df.groupby("County ")[["65-74", "75+"]].sum().sum(axis=1)
    age_df.columns = [0, 1, 2]

    # Create a vaccination rates dataframe
    vacc_rates = age_df / syn_pop_counts
    vacc_rates[["pop_0", "pop_1", "pop_2"]] = syn_pop_counts
    vacc_rates = vacc_rates.reset_index().rename(columns={"index": "County"})
    vacc_rates = vacc_rates[["County ", "pop_0", "pop_1", "pop_2", 0, 1, 2]]
    vacc_rates.columns = ["County", "pop_0", "pop_1", "pop_2", "reported_0", "reported_1", "reported_2"]

    for age in [0, 1, 2]:
        # Calculate Overall Vaccination Rate
        rate = age_df[age].sum() / syn_pop_counts[age].sum()
        print(f"Age: {age}, Vaccination Rate: {rate}")

        # Adjust the reported vaccination (since several columns are above 100%)
        vacc_rates.loc[vacc_rates[f"reported_{age}"] > rate, f"reported_{age}"] = rate
        current_rate = (vacc_rates[f"reported_{age}"] * vacc_rates[f"pop_{age}"]).sum() / vacc_rates[f"pop_{age}"].sum()
        ratio = rate / current_rate
        vacc_rates[f"adjusted_{age}"] = vacc_rates[f"reported_{age}"] * ratio

    vacc_rates.to_csv(filepaths["vaccination_rates_by_age"]["path"], index=False)
