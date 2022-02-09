import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import submodels.covid19.src.covid_data_input as cdi
from numba import njit
from tqdm import tqdm

import src.data_input as di

seir_params = {
    "length_infection": 6,  # ------------ Length of virus infection. source:
    "length_exposure": 5,  # ------------- Length of virus exposure before infection. source:
    "r0_estimate": 1.25,  # -------------- Estimated R0 of the virus. source:
    "immunity_length": 150,  # ----------- Number of days agents stay immune after getting the virus. source:
    "initial_case_multiplier": 10,  # ---- The case multiplier to use for earlier in the pandemic
    "middle_case_multiplier": 4,  # ------ The case multiplier to use for the middle of the pandemic
    "current_case_multiplier": 10,  # ------- The case multiplier to use for the forecast period
}


@njit()
def rollit(infections: np.array, days: int = 6):
    """ Find the estimated 'live infections' - This is the sum of the last `days` of cases"""
    infectious = np.zeros_like(infections)
    for i in range(len(infectious)):
        if i > days:
            infectious[i] = infections[(i - days) : i].sum()
        else:
            infectious[i] = 0
    return infectious


class NCSEIR:
    def __init__(self, seir_params: dict = seir_params) -> None:
        # Placeholders
        self.items = {}
        self.temp_df = pd.DataFrame()
        self.start_date = dt.date
        self.end_date = dt.date

        # Parameters
        self.seir_params = seir_params
        self.alpha = 1 / self.seir_params["length_exposure"]
        self.recovery_gamma = 1 / self.seir_params["length_infection"]
        self.recovery_tau = 1 / self.seir_params["immunity_length"]
        self.initial_beta = self.seir_params["r0_estimate"] / self.seir_params["length_infection"]

        # Data Prep
        self.county_data = {}
        self.county_population = di.nc_counties().set_index("County")["Population"].to_dict()
        self.prepare_data()

        # NSF Scenario Modeling
        self.county_adjustment = {}

    def prepare_data(self):
        """ Grab the most recent covid19 cases by county date available. Prepare the cases for modeling."""
        data = cdi.covid_cases()
        self.start_date = data.Date.min()
        self.end_date = data.Date.max()
        # Smooth Cases: Use a 10-day rolling window for smoothing
        data["Smooth_Cases"] = data.groupby("County")["New_Cases"].transform(lambda x: x.rolling(10, 1).mean())
        # Make sure Smooth = Total
        data["Smooth_Cases"] *= data["New_Cases"].sum() / data["Smooth_Cases"].sum()
        # Account for unreported cases, call this infections
        data["Infections"] = data.Date.map(self.multiplier_map) * data["Smooth_Cases"]
        # Cumulate Infections
        data["Cumulative_Infections"] = data.groupby("County")["Infections"].cumsum()
        data["Case_Multiplier"] = data.Date.map(self.multiplier_map)
        self.data = data
        for county, group in data.groupby(by=["County"]):
            self.county_data[county] = group.drop("County", axis=1).set_index("Date")
            self._prepare_count_data(county)

    def run_all(self, seir_start_date: dt.date, time_limit: int = 30, eff_r: float = 1.0):
        for county in tqdm(self.county_data.keys()):
            self.run_seir(county, seir_start_date, time_limit, eff_r)

        df = pd.concat([temp_df for key, temp_df in self.items.items()])
        self.temp_df = df
        return df

    @property
    def multiplier_map(self):
        """Under reporting of COVID cases is a known problem. We use 3 values to correct for this.
        - Initial case multiplier (icm): Used from 01/01/2020 through 06/01/2020
        - Middle case multiplier (mcm): Used from 06/01/2020-11/01/2021
        - Final case multiplier (fcm): Used from 11/01/2020 and beyond
        """
        icm = self.seir_params["initial_case_multiplier"]
        mcm = self.seir_params["middle_case_multiplier"]
        fcm = self.seir_params["current_case_multiplier"]
        t1 = self.start_date
        t2 = dt.date(2020, 6, 1)
        t3 = dt.date(2021, 12, 15)
        t4 = self.end_date + dt.timedelta(100)

        multiplier_map = {}
        days = (t2 - t1).days
        for i in range(days):
            multiplier_map[t1 + dt.timedelta(i)] = icm
        days = (t3 - t2).days
        for i in range(days):
            multiplier_map[t2 + dt.timedelta(i)] = mcm
        for i in range((t4 - t3).days + 1):
            multiplier_map[t3 + dt.timedelta(i)] = fcm
        return multiplier_map

    def graphic(self, county: str, graph_type: str, hide_actuals: bool = False):
        """[summary]

        Args:
            county (str): [description]
            graph_type (str): One of: "Infections", "Cases", "Cumulative Infections", "Cumulative Cases"
        Raises:
            ValueError: [description]
        """
        if county not in self.items:
            raise ValueError("Please run an SEIR model for this county before calling this function.")

        out_df = self.items[county]

        # Create random data with numpy
        x = out_df.index.values
        if graph_type == "Infections":
            y1 = out_df.Infections
            y2 = out_df.Est_Daily_Infections
        elif graph_type == "Cases":
            y1 = out_df.Smooth_Cases
            y2 = out_df.Est_Daily_Cases
        elif graph_type == "Cumulative Infections":
            y1 = out_df.Infections.cumsum()
            y2 = out_df.Est_Daily_Infections.cumsum()
        elif graph_type == "Cumulative Cases":
            y1 = out_df.Smooth_Cases.cumsum()
            y2 = out_df.Est_Daily_Cases.cumsum()

        if hide_actuals:
            y1 = y1[0 : (len(y1) - out_df.Projection.sum())]

        # Create traces
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y2, mode="lines", name=f"SEIR {graph_type}"))
        fig.add_trace(go.Scatter(x=x, y=y1, mode="lines", name=graph_type))
        fig.update_layout(title=f"{graph_type} for {county} County")
        plotly.offline.plot(fig, filename="temp.html")

    def _prepare_count_data(self, county: str):
        """Calculate the actual beta that occured during the COVID-19 Pandemic and the S,E,I, and R compartments"""
        # Filter to County
        county_data = self.county_data[county]

        # ----- Create a 7 day lag and calculate the growth rate
        county_data["Lag_Smooth_Cases"] = county_data["Smooth_Cases"].shift(7).fillna(0)
        county_data["Growth_Rate"] = (county_data["Smooth_Cases"] / county_data["Lag_Smooth_Cases"]) ** (1 / 7) - 1
        county_data.loc[county_data["Growth_Rate"].isna(), "Growth_Rate"] = 0

        # ----- Calculate the Re using formula from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1766383/
        gr = county_data["Growth_Rate"]
        county_data["Re"] = (1 + gr * seir_params["length_exposure"]) * (1 + gr * seir_params["length_infection"])

        # ----- Estimate the Susceptible at each day.
        county_pop = self.county_population[county]

        # People who are infected leave Susceptible - They return to Susceptible X Days Later
        leaving_s = county_data.Infections / county_pop
        join_s = leaving_s.shift(self.seir_params["immunity_length"]).fillna(0)
        county_data["Susceptible"] = 1 - leaving_s.cumsum() + join_s.cumsum()

        # ----- Estimate the Exposed at each day. Exposed a t is based on infected a t+1
        county_data["Exposed"] = (county_data["Infections"].shift(-1) / county_pop / self.alpha).fillna(0)
        # ----- Estimate the Infectious at each day
        days = self.seir_params["length_infection"]
        county_data["Infectious"] = rollit(county_data["Infections"].values, days) / county_pop
        # ----- Estimate Recovered
        county_data["Recovered"] = 1 - county_data[["Susceptible", "Exposed", "Infectious"]].sum(axis=1)

        # ----- Calculate Beta
        beta = (county_data["Re"] / seir_params["length_infection"]) / county_data["Susceptible"]
        county_data.insert(county_data.shape[1] - 4, "Beta", beta)
        # Beta is bounded by 0 and 1
        county_data.loc[(county_data["Beta"] < 0), "Beta"] = 0
        county_data.loc[(county_data["Beta"] > 1), "Beta"] = 1

    def run_seir(self, county: str, seir_start_date: dt.date, time_limit: int = 30, eff_r: float = 1.2):
        """Run the SEIRS-V Model"""
        # Setup Data
        county_data = self.county_data[county].copy()
        county_data_temp = county_data[county_data.index < seir_start_date]
        county_pop = self.county_population[county]

        # SEIR vectors
        susceptible = np.zeros(time_limit + 1)
        susceptible[0] = county_data_temp.Susceptible[-1]
        exposed = np.zeros(time_limit + 1)
        exposed[0] = county_data_temp.Exposed[-1]
        infected = np.zeros(time_limit + 1)
        infected[0] = county_data_temp.Infectious[-1]
        recovered = np.zeros(time_limit + 1)
        recovered[0] = county_data_temp.Recovered[-1]

        # Update beta based on input Re
        r0 = eff_r / susceptible[0]
        beta = r0 / self.seir_params["length_infection"]

        # Run SEIR
        for k in range(1, time_limit + 1):
            moving_out_s = beta * susceptible[k - 1] * infected[k - 1]
            moving_out_e = self.alpha * exposed[k - 1]
            moving_out_i = self.recovery_gamma * infected[k - 1]
            moving_out_r = self.recovery_tau * recovered[k - 1]
            susceptible[k] = susceptible[k - 1] - moving_out_s + moving_out_r
            exposed[k] = exposed[k - 1] + moving_out_s - moving_out_e
            infected[k] = infected[k - 1] + moving_out_e - moving_out_i
            recovered[k] = recovered[k - 1] + moving_out_i - moving_out_r

        out_df = pd.DataFrame(index=[seir_start_date + dt.timedelta(i) for i in range(time_limit)])
        out_df["Case_Multiplier"] = out_df.index.map(self.multiplier_map)
        out_df["Est_Live_Infections"] = np.round(infected[1:] * county_pop, 0)
        out_df["Est_Daily_Infections"] = np.round(self.alpha * exposed[1:] * county_pop, 0)
        out_df["Est_Daily_Cases"] = out_df["Est_Daily_Infections"] / out_df["Case_Multiplier"]
        out_df["Est_Cumulative_Infections"] = out_df["Est_Daily_Infections"].cumsum()

        temp_df = county_data.merge(out_df, how="outer", left_index=True, right_index=True)
        temp_df = temp_df.drop(["Case_Multiplier_x"], axis=1)
        temp_df = temp_df.rename(columns={"Case_Multiplier_y": "Case_Multiplier"})
        temp_df["County"] = county

        self.items[county] = temp_df[temp_df.index < seir_start_date + dt.timedelta(time_limit)].copy()

    # ----- NSF FUNCTIONS ----------------------------------------------------------------------------------------------
    def run_modified_seir(self, county: str, day: int = 1, length: int = 1, multiplier: float = 2, add_days: int = 0):
        county_data = self.county_data[county].copy()
        county_pop = self.county_population[county]

        if county in self.county_adjustment:
            beta_adjustment = self.county_adjustment[county]
        else:
            beta_adjustment = self.find_adjustment(county)

        if add_days > 0:
            new_data = pd.DataFrame(index=[county_data.index[-1] + dt.timedelta(i) for i in range(1, add_days + 1)])
            county_data = pd.concat([county_data, new_data])
            county_data.Beta = county_data.Beta.fillna(county_data.Beta[-(add_days + 1)])
            county_data.Case_Multiplier = county_data.Case_Multiplier.fillna(county_data.Beta[-(add_days + 1)])

        # Update Beta based on modifier
        if (day + length) > county_data.shape[0]:
            raise ValueError("Cannot run a modification that lasts longer than available data.")
        index = county_data.index[day : (day + length)]
        county_data.loc[index, "Beta"] *= multiplier
        # Bound Beta
        county_data.Beta[county_data.Beta < 0] = 0
        county_data.Beta[county_data.Beta > 1] = 1

        county_data = self.seir(county_data, beta_adjustment, county_pop)
        return county_data

    def seir(self, county_data, beta_adjustment, county_pop):

        susceptible = np.ones(county_data.shape[0])
        exposed = np.zeros(county_data.shape[0])
        infected = np.zeros(county_data.shape[0])
        recovered = np.zeros(county_data.shape[0])
        to_recovered = np.zeros(county_data.shape[0] + 1)
        beta = county_data.Beta.values * beta_adjustment

        days = county_data.shape[0]
        first_case = np.where(county_data.New_Cases > 0)[0][0]
        fca = first_case + 30

        # Let everything be the same for 30 days after the first case
        susceptible[0:fca] = county_data["Susceptible"][0:fca]
        exposed[0:fca] = county_data["Exposed"][0:fca]
        infected[0:fca] = county_data["Infectious"][0:fca]
        recovered[0:fca] = county_data["Recovered"][0:fca]

        for k in range(fca, days):
            moving_s_to_e = beta[k - 1] * susceptible[k - 1] * infected[k - 1]
            moving_e_to_i = self.alpha * exposed[k - 1]
            moving_i_to_r = self.recovery_gamma * infected[k - 1]
            to_recovered[k] = moving_i_to_r
            moving_r_to_s = 0
            if k > self.seir_params["immunity_length"]:
                moving_r_to_s = to_recovered[k - self.seir_params["immunity_length"]]
            susceptible[k] = susceptible[k - 1] - moving_s_to_e + moving_r_to_s
            exposed[k] = exposed[k - 1] + moving_s_to_e - moving_e_to_i
            infected[k] = infected[k - 1] + moving_e_to_i - moving_i_to_r
            recovered[k] = recovered[k - 1] + moving_i_to_r - moving_r_to_s

        county_data["Mod_Susceptible"] = susceptible
        county_data["Mod_Infectious"] = infected
        county_data["Mod_Exposed"] = exposed
        county_data["Mod_S_I"] = susceptible * infected
        county_data["Mod_Infections"] = self.alpha * county_data["Mod_Exposed"].shift(1) * county_pop
        county_data["Mod_Infections_Cumulative"] = county_data["Mod_Infections"].cumsum()
        county_data["Mod_Cases"] = county_data["Mod_Infections"] / county_data["Case_Multiplier"]
        county_data["Mod_Cases_Cumulative"] = county_data["Mod_Cases"].cumsum()

        return county_data

    def find_adjustment(self, county):
        county_data = self.county_data[county].copy()
        county_pop = self.county_population[county]

        error = 10 ** 10
        case_adjustment = 1
        for i in range(800, 1200, 5):
            i = i / 1000
            county_data = self.seir(county_data, i, county_pop)
            mse = ((county_data["Infections"] - county_data["Mod_Infections"]) ** 2).mean()
            if mse < error:
                case_adjustment = i
                error = mse
        self.county_adjustment[county] = case_adjustment
        return case_adjustment


if __name__ == "__main__":
    # A quick Example with modified SEIR
    self = NCSEIR(seir_params)
    # On day 100, we will increase transmisison by 5x for 10 days.
    county_data = self.run_modified_seir(county="Wake", day=100, length=10, multiplier=5, add_days=0)
    county_data["Mod_Cases"].plot()
    county_data["Smooth_Cases"].plot()
    plt.show()
