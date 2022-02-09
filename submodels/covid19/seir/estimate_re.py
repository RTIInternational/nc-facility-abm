# Calabrate an Re for COVID Cases in NC

import datetime as dt

import pandas as pd
import plotly
import plotly.graph_objects as go
from submodels.covid19.model.parameters import CovidParameters
from submodels.covid19.seir.seir import NCSEIR, seir_params

seir = NCSEIR(seir_params)
sd = dt.datetime.strptime(CovidParameters().start_date, "%Y-%m-%d").date()

rows = []
for r_effective in range(150, 250, 2):
    r_effective = r_effective / 100
    # Run the SEIR model
    df = seir.run_all(sd, 30, r_effective)
    # Compare the forecasted cases to the actual cases

    temp_df = df[df.index >= sd].reset_index()
    temp_df.to_csv("test.csv")

    cases = pd.DataFrame(temp_df.groupby("index").Smooth_Cases.sum())
    cases["Forecasted_Cases"] = temp_df.groupby("index").Est_Daily_Cases.sum()

    mape = abs(cases["Smooth_Cases"] - cases["Forecasted_Cases"]).mean()

    rows.append([r_effective, mape])

# Which Re was the closest for this time period?
rdf = pd.DataFrame(rows, columns=["r_effective", "mape"])
rdf.sort_values("mape")

best_re = 2.28
df = seir.run_all(sd, 30, best_re)
temp_df = df[df.index >= sd].reset_index()

cases = pd.DataFrame(temp_df.groupby("index").Smooth_Cases.sum())
cases["Forecasted_Cases"] = temp_df.groupby("index").Est_Daily_Cases.sum()

# Create random data with numpy
x = cases.index.values
y1 = cases.Smooth_Cases
y2 = cases.Forecasted_Cases

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y2, mode="lines", name="Forecasted Cases"))
fig.add_trace(go.Scatter(x=x, y=y1, mode="lines", name="Smoothed Reported Cases"))
fig.update_layout(title=f"Comparing Reported Case Counts for North Carolina to SEIR Forecasted Cases. Re={best_re}")
plotly.offline.plot(fig, filename="submodels/covid19/seir/compare_cases_to_forecasts.html")
