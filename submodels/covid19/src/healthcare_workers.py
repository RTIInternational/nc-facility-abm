from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from submodels.covid19.model.parameters import CovidParameters

main_filepaths = yaml.load(Path("config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)
submodel_filepaths = yaml.load(
    Path("submodels/covid19/config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader
)


def read_staffing_data() -> pd.DataFrame:
    """
    Reads Payroll Based Journal (PBJ) data (primary data source for hours worked by facility) and CareCompare data,
    (secondary data source for facilities missing PBJ data).

    Returns a single staffing dataframe, using PBJ data where available and CareCompare where not.
    """
    cms = pd.read_csv(main_filepaths["cms_data"]["path"])[
        [
            "federal_provider_number",
            "provider_name",
            "provider_county_name",
            "total_staff_hours_per_week",
            "County_Code",
        ]
    ]
    pbj = pd.read_csv(submodel_filepaths["pbj_processed"]["path"])

    merged = cms.merge(pbj, how="left", left_on="federal_provider_number", right_on="PROVNUM")

    merged.emp_hrs = np.where(np.isnan(merged.emp_hrs), merged.total_staff_hours_per_week, merged.emp_hrs)
    merged.ctr_hrs.fillna(0, inplace=True)
    merged.emp_hrs = np.round(merged.emp_hrs)
    merged.ctr_hrs = np.round(merged.ctr_hrs)

    return merged[
        ["federal_provider_number", "provider_name", "provider_county_name", "County_Code", "emp_hrs", "ctr_hrs"]
    ]


def read_county_distance_data(counties_to_include: list = None) -> pd.DataFrame:
    """
    Reads dataset containing distances from county to county in NC. This will be used in the worker assignment
    algorithm to assign workers to facilities close to them.
    """
    distances = pd.read_csv(main_filepaths["county_distances"]["path"])

    if counties_to_include:
        distances = distances.query("county_from in @counties_to_include and county_to in @counties_to_include")

    return distances


def calculate_target_worker_counts(
    staffing_data: pd.DataFrame, params: CovidParameters, multiplier: int = 1
) -> pd.DataFrame:
    """
    Uses hours worked per facility and assumptions based on parameters to calculate target worker counts.

    Outline of approach:

    - Assume constants defined in parameter file
    - Employee hours:
        - Assume all sites should have the same balance of single-site, part-time, and multi-site workers
        - Calculate how many hours in each category each site should have, using employee hours:
            - Hours worked by full-time, single-site workers
            - Hours worked by part-time, single-site workers
            - Hours worked as primary site by multi-site workers
            - Hours worked as secondary site by multi-site workers
        - Round these hours to the nearest unit of time workable by a worker of each category
    - Contract worker hours:
        - Round contarct worker hours to the nearest unit of time workable by a contract worker
    """

    # Use the constants to calculate how many hours should be worked in each category at each site.

    staffing_data["single_site_full_time_hrs"] = (
        staffing_data.emp_hrs
        * (1 - params.pct_multi_site_workers - (params.pct_part_time_workers * params.part_time))
        * multiplier
    )
    staffing_data["single_site_part_time_hrs"] = (
        staffing_data.emp_hrs * params.pct_part_time_workers * params.part_time * multiplier
    )
    staffing_data["multi_site_primary_hrs"] = (
        staffing_data.emp_hrs * params.pct_multi_site_workers * (1 - params.pct_time_second_site)
    ) * multiplier
    staffing_data["multi_site_secondary_hrs"] = (
        staffing_data.emp_hrs * params.pct_multi_site_workers * params.pct_time_second_site
    ) * multiplier

    def closest_divisible_by(dividend: float, divisor: int) -> int:
        """
        Finds the integer closest to the dividend that is divisible by the divisor.

        Used to round staffing hours to the nearest integer divisible by the unit of time worked by staff in
        that category.
        """
        remainder = dividend % divisor
        if remainder == 0:
            return dividend
        closest_lower = dividend - remainder
        closest_upper = dividend + divisor - remainder
        if dividend - closest_lower <= closest_upper - dividend:
            return closest_lower
        else:
            return closest_upper

    # Round the number of hours worked in each category to the nearest integer divisible by the unit of time worked
    # by staff in that category.
    #
    # For example, say we assume multi-site workers work 1/3 of their 39 hours = 13 at their secondary site.
    # Therefore, round each site's number of hours worked by multi-site workers as a secondary site to the nearest
    # integer divisible by 13.

    staffing_data["target_single_site_full_time_hrs"] = [
        closest_divisible_by(hrs, params.full_time_hours_per_week) for hrs in staffing_data["single_site_full_time_hrs"]
    ]
    staffing_data["target_single_site_part_time_hrs"] = [
        closest_divisible_by(hrs, params.full_time_hours_per_week * params.part_time)
        for hrs in staffing_data["single_site_part_time_hrs"]
    ]
    staffing_data["target_multi_site_primary_hrs"] = [
        closest_divisible_by(hrs, params.full_time_hours_per_week * (1 - params.pct_time_second_site))
        for hrs in staffing_data["multi_site_primary_hrs"]
    ]
    staffing_data["target_multi_site_secondary_hrs"] = [
        closest_divisible_by(hrs, params.full_time_hours_per_week * params.pct_time_second_site)
        for hrs in staffing_data["multi_site_secondary_hrs"]
    ]
    staffing_data["target_contract_hrs"] = [
        closest_divisible_by(hrs, params.full_time_hours_per_week / params.contract_worker_n_sites)
        for hrs in staffing_data["ctr_hrs"]
    ]

    # Convert target hours to target number of staff.
    #
    # These should all end up as integers due to the previous step. However, since PCT_TIME_SECOND_SITE and PART_TIME
    # are represented as floating point numbers, we round again here to account for any issues this may cause
    # (e.g. a site having 7.9999999999999 workers).

    staffing_data["target_single_site_full_time_workers"] = np.round(
        staffing_data.target_single_site_full_time_hrs / params.full_time_hours_per_week
    )
    staffing_data["target_single_site_part_time_workers"] = np.round(
        staffing_data.target_single_site_part_time_hrs / (params.full_time_hours_per_week * params.part_time)
    )
    staffing_data["target_multi_site_primary_workers"] = np.round(
        staffing_data.target_multi_site_primary_hrs
        / (params.full_time_hours_per_week * (1 - params.pct_time_second_site))
    )
    staffing_data["target_multi_site_secondary_workers"] = np.round(
        staffing_data.target_multi_site_secondary_hrs / (params.full_time_hours_per_week * params.pct_time_second_site)
    )
    staffing_data["target_contract_workers"] = np.round(
        staffing_data.target_contract_hrs / (params.full_time_hours_per_week / params.contract_worker_n_sites)
    )

    return staffing_data
