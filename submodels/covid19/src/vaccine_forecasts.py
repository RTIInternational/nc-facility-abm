import datetime as dt

import pandas as pd
import submodels.covid19.src.covid_data_input as cdi


def add_forecast(vaccines: pd.DataFrame, start_date: dt.date, days: int = 30):
    """Use the previous 2-week mean to predict the next 30 days of vaccines"""
    temp_df = vaccines[vaccines.Date < start_date].copy()
    weekly_mean = temp_df.groupby(by=["County"])["Daily_First_Doses"].apply(lambda x: x.iloc[-14:].mean()).to_dict()
    new_data = vaccines[vaccines.Date < start_date + dt.timedelta(days)].copy()
    new_data["Forecasted_Daily_First_Doses"] = round(new_data.County.map(weekly_mean)).astype(int)
    new_data.loc[(new_data.Date < start_date), "Forecasted_Daily_First_Doses"] = 0
    return new_data


def example():
    vaccines = cdi.vaccines()
    start_date = dt.date(2021, 8, 1)
    days = 30
    vaccines = add_forecast(vaccines, start_date, days)
    return vaccines
