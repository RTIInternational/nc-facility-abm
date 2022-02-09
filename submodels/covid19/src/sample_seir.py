import datetime as dt

import pandas as pd
import plotly
import plotly.graph_objects as go
from submodels.covid19.seir.seir import NCSEIR


def graph(df: pd.DataFrame, eff_r: float):
    df = df[df.index > dt.date(2021, 2, 28)]
    # Create random data with numpy
    x = df.index.values
    y1 = df.Cases

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines", name="Cases"))
    for column in df.columns:
        if column != "Cases":
            fig.add_trace(go.Scatter(x=x[-30:], y=df[column][-30:], mode="lines", name=column))
    fig.update_layout(title=f"Forecasted Case Counts for Wake County with and without Vaccines Modeled. Re = {eff_r}")
    fig.update_yaxes(range=[0, 400])
    plotly.offline.plot(fig, filename="temp.html")


if __name__ == "__main__":
    # Scenario 1 - Normal Parameters
    df = pd.DataFrame()
    names = ["With Vaccines", "Without Vaccines"]
    for eff_r in [0.5, 0.75, 1, 1.25]:
        for i, value in enumerate([True, False]):
            seir = NCSEIR(use_vaccines=value)
            start_date = dt.date(2021, 5, 1)
            seir.run_seir("Wake", start_date, time_limit=30, eff_r=eff_r)
            s1 = seir.items["Wake"]["Est_Daily_Cases"]
            df[names[i]] = s1
        df["Cases"] = seir.items["Wake"].Sm_Cases

        graph(df, eff_r)
