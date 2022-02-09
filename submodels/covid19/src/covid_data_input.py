import yaml
from pathlib import Path
import pandas as pd

filepaths = yaml.load(Path("submodels/covid19/config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)


def covid_cases():
    data = pd.read_csv(filepaths["covid_cases"]["path"])
    data.Date = [i.date() for i in pd.to_datetime(data.Date)]
    return data


def vaccines():
    vaccines = pd.read_csv(filepaths["vaccinations"]["path"])
    vaccines.Date = [i.date() for i in pd.to_datetime(vaccines.Date)]
    vaccines = vaccines.drop_duplicates(subset=["Date", "County"]).reset_index(drop=True)
    return vaccines


def vaccinations_by_age():
    vaccinations_by_age = pd.read_csv(filepaths["vaccinations_by_age"]["path"])
    return vaccinations_by_age


def vaccination_rates_by_age():
    vacc_rates = pd.read_csv(filepaths["vaccination_rates_by_age"]["path"])
    return vacc_rates.set_index("County")
