import numpy as np
import pandas as pd
from submodels.covid19.experiments.cohort_analysis.src.make_scenario import experiment_dir
from submodels.covid19.model.parameters import CovidParameters
from submodels.covid19.model.state import COVIDState

from src.misc_functions import get_multiplier


if __name__ == "__main__":
    """Compare the number of NH Admissions (Asymptomatic & Mild) to the number of HCWs going to work that are
    Asymptomatic or Mild, and the number of NH visitations that are Asymptomatic or Mild
    """

    scenario_dir = experiment_dir.joinpath("scenario_base")

    params = CovidParameters()
    params.update_from_file(scenario_dir.joinpath("parameters.yml"))
    multiplier = get_multiplier(params)

    nh_community_dfs = []
    nh_hospitals_dfs = []
    nh_visit_dfs = []
    nh_hcw_attendance_dfs = []

    for run in scenario_dir.glob("run_*"):
        # NH Admissions: From the Community
        admissions = pd.read_csv(run.joinpath("model_output/hospitalizations.csv"))
        nh_admissions = admissions[(admissions.Category == "NH") & (admissions.Location == 0)].copy()
        nh_admissions.rename(columns={"COVID_Status": "COVID_State"}, inplace=True)
        nh_abm_df = pd.DataFrame(nh_admissions.groupby(["COVID_State"]).size()) / multiplier
        nh_abm_df[0] = np.round(nh_abm_df[0]).astype(int)
        nh_abm_df.columns = [f"Count_{run.name}"]
        nh_community_dfs.append(nh_abm_df)

        # NH Admissions: From Hospitals
        nh_admissions = admissions[(admissions.Category == "NH") & (admissions.Location != 0)].copy()
        nh_admissions.rename(columns={"COVID_Status": "COVID_State"}, inplace=True)
        nh_abm_df = pd.DataFrame(nh_admissions.groupby(["COVID_State"]).size()) / multiplier
        nh_abm_df[0] = np.round(nh_abm_df[0]).astype(int)
        nh_abm_df.columns = [f"Count_{run.name}"]
        nh_hospitals_dfs.append(nh_abm_df)

        # NH Visits
        nh_visits = pd.read_csv(run.joinpath("model_output/nh_visits.csv"))
        nhdf = pd.DataFrame(nh_visits.groupby(["COVID_State"]).size()) / multiplier
        nhdf[0] = np.round(nhdf[0]).astype(int)
        nhdf.columns = [f"Count_{run.name}"]
        nh_visit_dfs.append(nhdf)

        # HCW
        hcw_attendance = pd.read_csv(run.joinpath("model_output/hcw_attendance.csv"))
        hcw_df = pd.DataFrame(hcw_attendance.groupby(["COVID_State"]).size()) / multiplier
        hcw_df[0] = np.round(hcw_df[0]).astype(int)
        hcw_df.columns = [f"Count_{run.name}"]
        nh_hcw_attendance_dfs.append(hcw_df)

    output_dir = experiment_dir.joinpath("analysis")
    output_dir.mkdir(exist_ok=True)

    columns = ["Community_to_NH", "Hospital_to_NH", "NH_Visits", "NH_HCWs"]
    dfs = []
    for i, df_list in enumerate([nh_community_dfs, nh_hospitals_dfs, nh_visit_dfs, nh_hcw_attendance_dfs]):
        nhdf = pd.concat(df_list, axis=1).reset_index()
        nhdf.fillna(0, inplace=True)
        nhdf["COVID_State"] = [COVIDState(i).name.title() for i in nhdf["COVID_State"]]
        nhdf.insert(1, f"{columns[i]}-Avg", nhdf[[i for i in nhdf.columns if "Count_" in i]].mean(axis=1))
        nhdf.insert(2, f"{columns[i]}-Std", nhdf[[i for i in nhdf.columns if "Count_" in i]].std(axis=1))
        nhdf = nhdf[[i for i in nhdf.columns if "Count" not in i]]
        dfs.append(nhdf)

    df = dfs[0].merge(dfs[1], left_on=["COVID_State"], right_on=["COVID_State"])
    for i in [2, 3]:
        df = df.merge(dfs[i], left_on=["COVID_State"], right_on=["COVID_State"])

    df.to_csv(output_dir.joinpath("nh_admissions_analysis.csv"), index=False)
