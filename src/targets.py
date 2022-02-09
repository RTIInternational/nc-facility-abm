import pandas as pd
import plotly.express as px

from model.hospital_abm import HospitalABM
from model.state import LifeState, LocationCategories

import src.data_input as di


def format_daily_state(model) -> pd.DataFrame:
    """Formats the daily counts array into a DataFrame
    Returns:
        pd.DataFrame: DataFrame indexed by (LIFE, LOCATION), columns=days.
    """
    days = list(range(model.params.time_horizon + 1))
    daily_state_numba_ndx = pd.MultiIndex.from_product(
        [[e.value for e in LifeState], list(model.nodes.facilities.keys()), days],
        names=["LIFE", "LOCATION", "day"],
    )
    daily_state = pd.DataFrame(model.daily_state_data.flatten(), index=daily_state_numba_ndx).unstack()
    daily_state.columns = days
    daily_state = daily_state.sort_index()
    return daily_state


def add_non_nc_to_daily_state(model, daily_state):
    for f_int in model.nodes.category_ints[LocationCategories.HOSPITAL.name]:
        non_nc = sum(model.nodes.facilities[f_int].agents == model.params.num_agents)
        daily_state.loc[(LifeState.ALIVE, f_int)] += non_nc
    return daily_state


def sum_movement(events, new: list = None, old: list = None):
    if old:
        events = events[events["Location"].isin(old)]
    if new:
        events = events[events["New_Location"].isin(new)]
    return events.shape[0]


class Analyze:
    def __init__(self, model: HospitalABM):
        self.model = model

        nh = di.nursing_homes()
        self.nh_fill_proportion = nh.average_number_of_residents_per_day.sum() / nh.Beds.sum()

        # User State
        self.life_events = self.model.life.state_changes.make_events()
        self.location_events = self.model.movement.location.state_changes.make_events()
        self.daily_state = format_daily_state(self.model)
        self.daily_state = add_non_nc_to_daily_state(self.model, self.daily_state)

        self.population_multiplier = 1 / self.model.multiplier
        self.time_multiplier = 365 / self.model.params.time_horizon
        self.compare_targets = self.make_targets()

    def make_targets(self):
        values = self.steady_state_values()
        values.extend(self.equilibrium_matrix())
        values.extend(self.count_deaths())

        df = pd.DataFrame(values)
        df.columns = ["Description", "Target", "Value"]
        df.Target = df.Target.astype(int)
        df.Value = df.Value.astype(int)
        df["MAPE"] = abs(df.Target - df.Value) / df.Target
        df.MAPE = df.MAPE.fillna(0) * 100

        df.to_csv(self.model.output_dir.joinpath("targets.csv"), index=False)
        return df

    def steady_state_values(self) -> list:
        # Targets
        population = self.model.params.nc_population
        steady_states = {}
        df = self.model.hospital_df
        hospital_total = df.Acute_Beds * self.model.params.location.acute_fill_proportion
        hospital_total += df.ICU_Beds * self.model.params.location.icu_fill_proportion
        hospital_total = hospital_total.sum()
        steady_states[LocationCategories.HOSPITAL.name] = hospital_total
        steady_states[LocationCategories.NH.name] = di.nursing_homes().Beds.sum() * self.nh_fill_proportion
        steady_states[LocationCategories.LT.name] = (
            di.ltachs().Beds.sum() * self.model.params.location.lt_fill_proportion
        )
        steady_states[LocationCategories.COMMUNITY.name] = population - sum(list(steady_states.values()))

        values = []
        for category in self.model.nodes.categories:
            locations = self.model.nodes.category_ints[category]
            data = self.facility_population(locations).sum()
            if data.shape[0] > 0:
                values.append(
                    [
                        f"Steady State: {category}",
                        steady_states[category],
                        data.mean() * self.population_multiplier,
                    ]
                )
            else:
                values.append([f"Steady State: {category}", steady_states[category], 0])

        return values

    def equilibrium_matrix(self) -> list:
        # How many people leave each node, and where do they go?
        rows = []
        for old_cat in self.model.nodes.categories:
            old_nodes = self.model.nodes.category_ints[old_cat]
            values = []
            for new_cat in self.model.nodes.categories:
                new_nodes = self.model.nodes.category_ints[new_cat]
                values.append(sum_movement(self.location_events, new=new_nodes, old=old_nodes))
            rows.append(values)

        df = pd.DataFrame(rows)
        df.index = self.model.nodes.categories
        df.columns = self.model.nodes.categories
        df = df.multiply(self.population_multiplier * self.time_multiplier)
        fbf = self.model.four_by_four.copy()
        fbf["COMMUNITY"] += fbf["Death"]
        df_targets = fbf.drop("Death", axis=1)
        values = []
        for category in self.model.nodes.categories:
            for category2 in self.model.nodes.categories:
                v = df.loc[category].loc[category2]
                t = df_targets.loc[category].loc[category2]
                values.append([f"{category} -> {category2}", t, v])
        return values

    def count_deaths(self) -> list:
        le = self.life_events
        death_targets = self.model.four_by_four["Death"]
        values = []
        for category in self.model.nodes.categories:
            locations = self.model.nodes.category_ints[category]
            v = int(le[le.Location.isin(locations)].shape[0]) * self.population_multiplier * self.time_multiplier
            t = death_targets.loc[category]
            values.append([f"Deaths: {category}", t, v])
        return values

    def facility_population(self, locations: list = None) -> pd.DataFrame:
        dc = self.daily_state.reset_index()
        dc = dc[dc.LIFE == LifeState.ALIVE.value]
        if locations:
            dc = dc[dc["LOCATION"].isin(locations)]
        return dc.drop(["LIFE", "LOCATION"], axis=1)

    def location_capacity(self, category: str) -> pd.DataFrame:
        locations = self.model.nodes.category_ints[category]
        temp_df = self.facility_population(locations=locations)
        temp_df *= self.population_multiplier
        df = pd.DataFrame(temp_df.mean(axis=1).round(2), columns=["Avg Capacity"])
        df["Min"] = temp_df.min(axis=1)
        df["Max"] = temp_df.max(axis=1).astype(int)
        df["Sd."] = temp_df.std(axis=1).astype(int)
        df["Facility_Name"] = [self.model.nodes.facilities[i].name for i in locations]
        df.set_index("Facility_Name", inplace=True)
        df["Facility ID"] = locations

        if category == LocationCategories.HOSPITAL.name:
            df = df.merge(self.model.hospital_df, left_index=True, right_index=True)
            df["Expected Capacity"] = df[["Acute_NC_Agents", "Acute_Non_NC_Agents"]].sum(axis=1)
            df["Expected Capacity"] += df[["ICU_NC_Agents", "ICU_Non_NC_Agents"]].sum(axis=1)

        if category == LocationCategories.NH.name:
            nh = di.nursing_homes().set_index("Name")
            df = df.merge(nh, left_index=True, right_index=True)
            df["Expected Capacity"] = round(df["Beds"] * self.nh_fill_proportion, 0).astype(int)

        if category == LocationCategories.LT.name:
            lt = di.ltachs().set_index("Name")
            df = df.merge(lt, left_index=True, right_index=True)
            df["Expected Capacity"] = round(df["Beds"] * self.model.params.location.lt_fill_proportion, 0).astype(int)

        df["Modeled Over Expected"] = df["Avg Capacity"] / df["Expected Capacity"]
        df = df.sort_values(by=["Avg Capacity"], ascending=False)
        df[["Facility ID", "Avg Capacity", "Min", "Max", "Sd.", "Expected Capacity"]].to_csv(
            self.model.output_dir.joinpath(f"{category}_capacity.csv"), index=False
        )
        return df

    def facility_los(self, category: str) -> pd.DataFrame:
        """Looking at LOS values for a specific category, only consider events after day 60.
        The first several days of the model will also include some "initial LOS" values. We don't want to include these.

        """
        le = self.location_events
        events = self.location_events[self.location_events.Location.isin(self.model.nodes.category_ints[category])]
        events = events[events.Time > 30]
        facility_ids = list(events.Location.value_counts().index)

        results = []
        for facility_id in facility_ids:
            admissions = le[le.New_Location == facility_id]
            discharges = le[le.Location == facility_id]

            v1 = int(admissions.shape[0] / self.model.multiplier)

            v2 = discharges.LOS.mean().round(2)
            v3 = discharges.LOS.std().round(2)
            if category == "HOSPITAL":
                values = pd.Series(model.movement.los[facility_id])
            else:
                values = pd.Series(self.model.movement.los[category])
            t2 = values.mean().round(2)
            t3 = values.std().round(2)

            results.append([facility_id, v1, v2, t2, v3, t3])

        df = pd.DataFrame(
            results,
            columns=[
                "Facility ID",
                "Admissions",
                "Avg LOS",
                "Exp. Avg LOS",
                "Sd. LOS",
                "Exp. Sd. LOS",
            ],
        )
        df.to_csv(self.model.output_dir.joinpath(f"{category}_los.csv"), index=False)
        return df

    def graph_steady_states(self):

        df = pd.DataFrame()
        for category in self.model.nodes.categories:
            locations = self.model.nodes.category_ints[category]
            df[category] = self.facility_population(locations).sum().values
        df.reset_index(inplace=True)

        df = pd.melt(df, id_vars="index", value_vars=df.columns[1:])
        df.columns = ["Day", "Category", "Count"]

        fig = px.line(df, x="Day", y="Count", color="Category", title="")
        fig.show()

    def graph_movement(self, from_category: LocationCategories = None, to_category: LocationCategories = None):
        le = self.location_events
        if from_category:
            le = le[le.Location.isin(self.model.nodes.category_ints[from_category])]
        if to_category:
            le = le[le["New_Location"].isin(self.model.nodes.category_ints[to_category])]

        df = pd.DataFrame(le.groupby("Time").size())
        df.reset_index(inplace=True)
        df.columns = ["Day", "Count"]
        fig = px.line(df, x="Day", y="Count", title="")
        fig.show()
        return df


if __name__ == "__main__":

    """This script will generate the required output for the ODD "Patterns" section."""
    model = HospitalABM(scenario_dir="experiments/base/scenario_base_full", run_dir="run_0", seed=1111)
    model.run_model(True)

    analysis = Analyze(model)

    # ----- Pattern 1: Length of Stay
    analysis.facility_los("HOSPITAL")
    analysis.facility_los("LT")
    # NH LOS is too long to consider. For example - running a model for 365 days will never capture individuals that
    # stay for 1-3 years. The produced average LOS will always be much smaller than reality.
    # analysis.facility_los("NH")

    # ----- Pattern 2: Average Capacity
    analysis.location_capacity("HOSPITAL")
    analysis.location_capacity("LT")
    analysis.location_capacity("NH")

    # ----- Pattern 3: Patient Movement Between Facilities
    analysis.compare_targets
