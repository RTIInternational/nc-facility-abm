import pandas as pd
import plotly.express as px
from submodels.covid19.model.covid import COVIDModel
from submodels.covid19.model.state import COVIDState, VaccinationStatus

from src.targets import Analyze


class AnalyzeCOVID(Analyze):
    def __init__(self, model: str):
        super().__init__(model)
        pass

    def pattern_one(self, county: int = None):
        """We want to look at the number of new cases in each county compared the expected number of cases

        Given a county id, look at the expected cases per day compared to the actual cases
        """
        sd = pd.to_datetime(self.model.start_date)

        # Expected Potential Cases
        cases = self.model.cases.reset_index().copy()
        cases = cases[(cases.Date > sd)]
        if county:
            county_str = self.model.county_codes_dict[county]
            cases = cases[cases.County == county_str]
        else:
            cases = cases.groupby("Date").sum().reset_index()

        # Actual Cases
        model_cases = self.model.covid_cases.make_events()
        model_cases["County"] = model_cases.Unique_ID.apply(lambda x: self.model.county_code[x])
        if county:
            model_cases = model_cases[model_cases.County == county].groupby("Time").size() * (1 / self.model.multiplier)
        else:
            model_cases = model_cases.groupby("Time").size() * (1 / self.model.multiplier)

        new_df = cases[["Date", "Est_Daily_Infections"]].copy().reset_index(drop=True)
        new_df.columns = ["Date", "Expected Daily Infections"]
        new_df["Modeled Infections"] = model_cases.values[1:].round(0).astype(int)
        new_df = pd.melt(new_df, id_vars="Date", value_vars=new_df.columns[1:])
        new_df.columns = ["Date", "Category", "Cases"]

        fig = px.line(
            new_df,
            x="Date",
            y="Cases",
            color="Category",
            title="Modeled vs. Expected Infections: Pattern One",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(
            font_family="Times New Roman",
        )
        fig.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
            }
        )
        fig.update_layout(xaxis=dict(tickformat="%m-%d-%Y"))
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black")
        fig.update_yaxes(rangemode="tozero")
        fig.show()

    def pattern_two(self):
        """ Compare the percentage of cases that turned out to be hospitalizations by age and vaccine status"""
        # We need case counts by vaccination status
        for item in [True, False]:
            cases = self.model.covid_cases.make_events()
            # Do not include time 0. This is not part of the simulation and is set by an input parameter only
            cases = cases[cases.Time != 0]
            cases = cases[cases.Reported == item]
            cases["Age"] = [model.age_group[i] for i in cases.Unique_ID]
            # Find the proportion of outcome by vaccination status
            mc = pd.DataFrame(cases.groupby(["Vacc_Status", "Age"])["Type"].value_counts(1))
            mc.columns = ["Proportion"]
            mc = mc.reset_index()
            # Add the total number of events
            mc["Cases"] = cases.groupby(["Vacc_Status", "Age"])["Type"].value_counts().values
            mc = mc.sort_values(by=["Vacc_Status", "Type"])
            # Mape the values to their names
            vmap = {1: "Not Vaccinated", 2: "Vaccinated"}
            mc.Vacc_Status = mc.Vacc_Status.map(vmap)
            mc.Type = mc.Type.map(lambda x: COVIDState(x).name).str.title()
            mc.columns = ["Vaccination Status", "Age", "COVID State", "Modeled Proportion", "Cases"]
            mc = mc.sort_values(by=["Vaccination Status", "Age"])

            reported = "not_reported"
            if item:
                reported = "reported"
            targets = []
            for _, row in mc.iterrows():
                if row["Vaccination Status"] == "Vaccinated":
                    vacc_status = VaccinationStatus.VACCINATED
                else:
                    vacc_status = VaccinationStatus.NOTVACCINATED
                age_group = row["Age"]
                v = self.model.covid_distributions[reported][vacc_status][age_group]
                if row["COVID State"] == COVIDState.ASYMPTOMATIC.name.title():
                    targets.append(v[0])
                if row["COVID State"] == COVIDState.MILD.name.title():
                    targets.append(v[1] - v[0])
                if row["COVID State"] == COVIDState.SEVERE.name.title():
                    targets.append(v[2] - v[1])
                if row["COVID State"] == COVIDState.CRITICAL.name.title():
                    targets.append(v[3] - v[2])

            mc["Target Proportion"] = targets
            mc = mc[["Vaccination Status", "Age", "COVID State", "Cases", "Modeled Proportion", "Target Proportion"]]
            mc.to_csv(self.model.output_dir.joinpath(f"pattern_2_{reported}.csv"), index=False)

    def pattern_three(self):
        pass


if __name__ == "__main__":

    """This script will generate the required output for the ODD "Patterns" section."""
    model = COVIDModel(scenario_dir="submodels/covid19/experiments/base/scenario_base", run_dir="run_0", seed=1111)
    model.run_model()

    analysis = AnalyzeCOVID(model)
    analysis.pattern_one(135)

    analysis.pattern_two()

    analysis.pattern_three()
