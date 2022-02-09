import pandas as pd
import numpy as np
import fire

from kats.consts import TimeSeriesData
from kats.detectors.outlier import OutlierDetector
from kats.models.prophet import ProphetModel, ProphetParams

import yaml
from pathlib import Path


def reverse_cumsum(df: pd.DataFrame, var: str) -> pd.Series:
    """
    Calculates daily counts from a cumulative sum column.
    """
    counties = df.County.unique()
    dfs = []

    for county in counties:
        dff = df.query("County == @county").copy()
        dff.sort_values("Date", inplace=True)
        dff["reversed_cumsum"] = dff[var].diff()
        dfs.append(dff)

    df = pd.concat(dfs).sort_values(["County", "Date"])
    return df["reversed_cumsum"]


def clean(vax: pd.DataFrame) -> pd.DataFrame:
    """
    Filters national vaccinations by county down to NC only, then appplies various
    data cleaning steps.
    """
    vax_nc = vax.query("Recip_State == 'NC'").copy().reset_index(drop=True)
    vax_nc.Recip_County = vax_nc.Recip_County.str.replace(" County", "")
    vax_nc.rename(columns={"Recip_County": "County", "FIPS": "County_FIPS"}, inplace=True)
    vax_nc.drop(columns=["MMWR_week", "Recip_State"], inplace=True)
    vax_nc.fillna(0, inplace=True)
    vax_nc["Date"] = pd.to_datetime(vax_nc["Date"])
    vax_nc.sort_values(["County", "Date"], inplace=True)
    vax_nc["first_doses_administered"] = reverse_cumsum(vax_nc, "Administered_Dose1_Recip")
    vax_nc = vax_nc[["Date", "County", "County_FIPS", "first_doses_administered"]]
    return vax_nc


def filter_df_to_county(df: pd.DataFrame, county: str) -> pd.DataFrame:
    """
    Filters dataframe to a single county
    """
    return df.query("County == @county").copy().reset_index()


def prepare_df_for_forecast(df: pd.DataFrame, forecast_start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Prepares one county's vaccination data for forecasting.
    """
    # remove days with zero vaccinations before first day of vaccinations
    # omit first day of vaccinations because it is artificially high for all counties
    start_index = min(df.index[df.first_doses_administered > 0]) + 1
    dff = df.iloc[start_index:, :].copy().reset_index()
    # remove any days of data after forecast start date
    dff = dff.query("Date < @forecast_start_date")
    # forecast will use log transformed data to add lower bound of zero
    # we'll transform back after forecasting
    # replace zeros with ones to avoid errors in log transform
    dff.first_doses_administered = np.where(
        dff.first_doses_administered.eq(0),
        1,
        dff.first_doses_administered,
    )
    dff["first_doses_log"] = np.log(dff.first_doses_administered)
    dff.first_doses_log = np.where(dff.first_doses_log.eq(0), 1, dff.first_doses_log)
    return dff


def remove_outliers(ts: TimeSeriesData) -> TimeSeriesData:
    """
    Identifies outliers in a time series and removes them with interpolation.
    """
    ts_outlierDetection = OutlierDetector(ts, "multiplicative")
    ts_outlierDetection.detector()
    return ts_outlierDetection.remover(interpolate=True)


def forecast_with_prophet(ts: TimeSeriesData, days_to_forecast: int) -> pd.DataFrame:
    """
    Fits a Prophet model to a log-transformed time series and generates a 30-day
    forecast. Exponentiates the resulting forecast to get it back in the original
    scale.
    """
    params = ProphetParams(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        floor=0,
    )

    m = ProphetModel(ts, params)
    m.fit()
    forecast = m.predict(days_to_forecast, floor=0)

    for var in ["fcst", "fcst_lower", "fcst_upper"]:
        forecast[var] = np.exp(forecast[var])

    return forecast


def add_forecast_to_df(df: pd.DataFrame, forecast: pd.DataFrame, county: str, fips: int) -> pd.DataFrame:
    """
    Concatenates a county's vaccination data with its forecasted vaccinations.
    """
    forecast.rename(
        columns={
            "time": "Date",
            "fcst": "first_doses_administered",
            "fcst_lower": "first_doses_administered_lower",
            "fcst_upper": "first_doses_administered_upper",
        },
        inplace=True,
    )

    forecast["forecast"] = True
    forecast["County"] = county
    forecast["County_FIPS"] = fips

    df["forecast"] = False

    return df.append(forecast)


def forecast_all_counties(df: pd.DataFrame, forecast_start_date: pd.Timestamp, days_to_forecast: int) -> pd.DataFrame:
    """
    Loops through all counties in the dataset and generates forecasts for each.
    Combines all forecasts in a single dataframe.
    """
    dfs_with_forecasts = []

    for county in df.County.unique():
        print("forecasting for:", county)

        vax_county = filter_df_to_county(df, county)
        fips = vax_county.County_FIPS[0]
        vax_county_for_forecast = prepare_df_for_forecast(vax_county, forecast_start_date=forecast_start_date)
        ts = TimeSeriesData(
            time=vax_county_for_forecast.Date,
            value=vax_county_for_forecast.first_doses_log,
        )
        ts_outliers_interpolated = remove_outliers(ts)
        forecast = forecast_with_prophet(ts_outliers_interpolated, days_to_forecast=days_to_forecast)
        df_with_forecast = add_forecast_to_df(vax_county, forecast, county, fips)
        dfs_with_forecasts.append(df_with_forecast)

    combined_forecasts = pd.concat(dfs_with_forecasts)
    if "index" in combined_forecasts.columns:
        combined_forecasts.drop(columns=["index"], inplace=True)
    return combined_forecasts


def main(forecast_start_date: str = "2021-07-22", days_to_forecast: int = 30):
    filepaths = yaml.load(
        Path("submodels/covid19/config/filepaths.yaml").read_text(),
        Loader=yaml.loader.SafeLoader,
    )
    inpath = Path(filepaths["vaccinations_raw"]["path"])
    vax = pd.read_csv(inpath)

    vax_nc = clean(vax)

    vax_nc_with_forecasts = forecast_all_counties(
        vax_nc,
        forecast_start_date=pd.Timestamp(forecast_start_date),
        days_to_forecast=days_to_forecast,
    )

    outpath = Path(filepaths["vaccinations_processed"]["path"])
    vax_nc_with_forecasts.to_csv(outpath, index=False)


if __name__ == "__main__":
    fire.Fire(main)
