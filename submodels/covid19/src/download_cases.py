import argparse
import io
from pathlib import Path

import pandas as pd
import requests
import yaml

filepaths = yaml.load(Path("submodels/covid19/config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("key", type=str, help="API Key for COVID Act Now")
    args = parser.parse_args()

    response = requests.get(f"https://api.covidactnow.org/v2/county/NC.timeseries.csv?apiKey={args.key}")
    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    df = df[["county", "date", "actuals.cases", "actuals.newCases"]].copy()
    df.columns = ["County", "Date", "Cumulative_Cases", "New_Cases"]
    df.County = df.County.str.replace(" County", "")
    df = df.fillna(0)

    df.to_csv(filepaths["covid_cases"]["path"], index=False)
