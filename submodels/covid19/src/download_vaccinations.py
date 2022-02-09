from pathlib import Path

import pandas as pd
import yaml
from sodapy import Socrata


filepaths = yaml.load(Path("submodels/covid19/config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)


if __name__ == "__main__":
    client = Socrata("data.cdc.gov", None)

    # Only grab needed columns for NC
    columns = "date, recip_county, administered_dose1_recip"
    results = client.get_all("8xkx-amqh", recip_state="NC", select=columns)
    results_df = pd.DataFrame.from_records(results)

    # Clean up the data and rename columns
    results_df.recip_county = results_df.recip_county.str.replace(" County", "")
    results_df.date = results_df.date.str[0:10]
    results_df.columns = ["Date", "County", "First_Doses_Administered"]

    # Sort by Date and County
    results_df = results_df.sort_values(by=["Date", "County"]).reset_index(drop=True).fillna(0)

    # Find the number of first doses by day
    results_df.First_Doses_Administered = results_df.First_Doses_Administered.astype(int)
    results_df["Daily_First_Doses"] = results_df.groupby(["County"])["First_Doses_Administered"].diff().fillna(0)

    # Drop Unknown
    results_df = results_df[results_df.County != "Unknown"]

    # Save File
    results_df.to_csv(filepaths["vaccinations"]["path"], index=False)
