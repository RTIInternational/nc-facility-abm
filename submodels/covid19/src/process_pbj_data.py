from copy import copy
from pathlib import Path

import pandas as pd
import yaml


def get_weekly_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    PBJ data has hours worked by staff type per day. We want hours worked by week to calculate worker counts.

    There are many possible ways to approach this. Here, we calculate the average hours per day across the
    full quarter of daily counts. We then multiply that value by 7 for a weekly average.

    In the PBJ data, employee variables are noted with the prefix "emp_" and contract worker variables are noted
    with the prefix "_ctr".
    """

    def get_col_lists(df):
        emp = [i for i in list(df.columns) if i.endswith("emp")]
        ctr = [i for i in list(df.columns) if i.endswith("ctr")]
        both = copy(emp)
        both.extend(ctr)
        return emp, ctr, both

    emp, ctr, both = get_col_lists(df)

    grouped = df.groupby("PROVNUM").agg("mean")
    grouped = grouped[both].reset_index()
    grouped["emp_hrs"] = grouped[emp].sum(axis=1) * 7
    grouped["ctr_hrs"] = grouped[ctr].sum(axis=1) * 7
    grouped["total_hrs"] = grouped["emp_hrs"] + grouped["ctr_hrs"]
    grouped["emp_ctr_ratio"] = grouped["emp_hrs"] / grouped["ctr_hrs"]
    grouped = grouped[["PROVNUM", "emp_hrs", "ctr_hrs", "total_hrs", "emp_ctr_ratio"]]

    return grouped


def main():
    filepaths = yaml.load(Path("submodels/covid19/config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)

    pbj_nurse = pd.read_csv(filepaths["pbj_nurse"]["path"], encoding="ISO-8859-1").query("STATE == 'NC'")
    pbj_non = pd.read_csv(filepaths["pbj_non_nurse"]["path"], encoding="ISO-8859-1").query("STATE == 'NC'")

    pbj_nurse.PROVNUM = pbj_nurse.PROVNUM.astype(str)
    pbj_non.PROVNUM = pbj_non.PROVNUM.astype(str)

    pbj_nurse_weekly = get_weekly_hours(pbj_nurse)
    pbj_non_weekly = get_weekly_hours(pbj_non)

    pbj = pbj_nurse_weekly.merge(pbj_non_weekly, how="inner", on="PROVNUM", suffixes=("_nurse", "_non"))
    pbj["emp_hrs"] = pbj["emp_hrs_non"] + pbj["emp_hrs_nurse"]
    pbj["ctr_hrs"] = pbj["ctr_hrs_non"] + pbj["ctr_hrs_nurse"]
    pbj["total_hrs"] = pbj["emp_hrs"] + pbj["ctr_hrs"]

    pbj.to_csv(filepaths["pbj_processed"]["path"], index=False)


if __name__ == "__main__":
    main()
