from pathlib import Path
import pandas as pd
from submodels.covid19.model.state import COVIDState, VaccinationStatus
import numpy as np
from src.misc_functions import get_multiplier
from submodels.covid19.model.parameters import CovidParameters
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly as plotly

experiment_dir = Path("submodels/covid19/experiments/nsf_01_2021")


def base_graphic(base_df, base_cases_df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Left Side
    for value in ["Asymptomatic", "Mild", "Hospitalized for COVID-19"]:
        if value == "Hospitalized for COVID-19":
            temp_df = base_df[base_df.COVID_Status.isin(["Severe", "Critical"])]
            avg = temp_df.groupby("Day")["Average"].sum()
        else:
            temp_df = base_df[base_df.COVID_Status == value]
            avg = temp_df.Average
        fig.add_trace(
            go.Scatter(x=temp_df.Day, y=avg, name=f"{value}"),
            secondary_y=False,
        )

    # Add Right Side
    temp_df = base_cases_df[base_cases_df.index != 0]
    fig.add_trace(
        go.Scatter(x=temp_df.index, y=temp_df.Average, name="COVID Infections Per 100k"),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(title_text="Average Hospital Admissions Per Day & Average Infections Per Day Per 100,000 People")

    # Set x-axis title
    fig.update_xaxes(title_text="Simulation Day")

    # Set y-axes titles
    fig.update_yaxes(title_text="Average Admissions", secondary_y=False)
    fig.update_yaxes(title_text="Average Modeled Infections Per Day Per 100,000 People", secondary_y=True)

    plotly.offline.plot(fig, filename=str(output_dir.joinpath("cases_over_time_v2.html")))


if __name__ == "__main__":

    selected_hospital_id = None
    for scenario_dir in experiment_dir.glob("scenario*"):
        params = CovidParameters()
        params.update_from_file(scenario_dir.joinpath("parameters.yml"))
        multiplier = get_multiplier(params)

        hospitalizations_dfs = []
        individual_hospital_dfs = []
        base_dfs = []
        base_cases = []

        for run in scenario_dir.glob("run_*"):
            # Get the hospitalizations for each run
            try:
                df = pd.read_csv(f"{run}/model_output/hospitalizations.csv")
            except FileNotFoundError:
                continue

            # Count the number of entries by COVID status
            df = df[df.Category == "HOSPITAL"]
            th = pd.DataFrame(df.groupby(["COVID_Status", "Vaccination_Status"]).size() / multiplier)
            th[0] = np.round(th[0]).astype(int)
            th.columns = [f"Count_{run.name}"]
            hospitalizations_dfs.append(th)

            if "base" in str(scenario_dir):
                try:
                    cases = pd.read_csv(f"{run}/model_output/covid_cases.csv")
                except FileNotFoundError:
                    continue
                temp_df = pd.DataFrame(df.groupby(["COVID_Status", "Time"]).size() / multiplier)
                temp_df.columns = [f"Count_{run.name}"]
                base_dfs.append(temp_df)

                # Read Cases
                avg_cases = cases.groupby("Time").size() / params.num_agents * 100000
                base_cases.append(avg_cases)

            # Count the number for one hospital
            if selected_hospital_id:
                pass
            else:
                selected_hospital_id = df.New_Location.value_counts().index[0]

            temp_df = df[df.New_Location == selected_hospital_id]
            temp_df = pd.DataFrame(temp_df.groupby(["COVID_Status", "Vaccination_Status"]).size() / multiplier)
            temp_df[0] = np.round(temp_df[0]).astype(int)
            temp_df.columns = [f"Count_{run.name}"]
            individual_hospital_dfs.append(temp_df)

        output_dir = experiment_dir.joinpath("analysis")
        output_dir.mkdir(exist_ok=True)

        # ----- Hospitalizations by Category
        hdf = pd.concat(hospitalizations_dfs, axis=1).reset_index()
        hdf["COVID_Status"] = [COVIDState(i).name.title() for i in hdf["COVID_Status"]]
        hdf["Vaccination_Status"] = [VaccinationStatus(i).name.title() for i in hdf["Vaccination_Status"]]
        hdf.insert(2, "Average", hdf[[i for i in hdf.columns if "Count_" in i]].mean(axis=1))
        hdf.insert(3, "Std", hdf[[i for i in hdf.columns if "Count_" in i]].std(axis=1))
        hdf.insert(4, "Max", hdf[[i for i in hdf.columns if "Count_" in i]].max(axis=1))
        hdf.insert(5, "Min", hdf[[i for i in hdf.columns if "Count_" in i]].min(axis=1))
        hdf.to_csv(output_dir.joinpath(f"{scenario_dir.name}-hospitalizations_by_covid_state.csv"))

        # ----- Hospitalizations Over Time for base
        if "base" in str(scenario_dir):
            base_df = pd.concat(base_dfs, axis=1)
            base_df = base_df.reset_index()
            base_df["COVID_Status"] = [COVIDState(i).name.title() for i in base_df["COVID_Status"]]
            base_df.insert(0, "Average", base_df[[i for i in base_df.columns if "Count_" in i]].mean(axis=1))
            base_df = base_df.rename(columns={"Time": "Day"})
            base_df.to_csv(output_dir.joinpath(f"{scenario_dir.name}-admissions_by_time.csv"))

            base_cases_df = pd.concat(base_cases, axis=1).reset_index(drop=True)
            base_cases_df.insert(0, "Average", base_cases_df.mean(axis=1))
            base_cases_df.to_csv(output_dir.joinpath(f"{scenario_dir.name}-cases-per-100k.csv"))
            base_graphic(base_df, base_cases_df)

        # ----- Hospitalizations for One Specific Hospital
        ihdf = pd.concat(individual_hospital_dfs, axis=1).reset_index()
        ihdf["COVID_Status"] = [COVIDState(i).name.title() for i in ihdf["COVID_Status"]]
        ihdf["Vaccination_Status"] = [VaccinationStatus(i).name.title() for i in ihdf["Vaccination_Status"]]
        ihdf.insert(2, "Average", ihdf[[i for i in ihdf.columns if "Count_" in i]].mean(axis=1))
        ihdf.insert(3, "Std", ihdf[[i for i in ihdf.columns if "Count_" in i]].std(axis=1))
        ihdf.insert(4, "Max", ihdf[[i for i in ihdf.columns if "Count_" in i]].max(axis=1))
        ihdf.insert(5, "Min", ihdf[[i for i in ihdf.columns if "Count_" in i]].min(axis=1))
        ihdf.to_csv(output_dir.joinpath(f"{scenario_dir.name}-one_hospital_by_covid_state.csv"))
