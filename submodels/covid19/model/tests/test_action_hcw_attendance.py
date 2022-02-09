from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from model.facilities import NursingHome
from submodels.covid19.src.healthcare_workers import calculate_target_worker_counts, read_staffing_data


@pytest.mark.skip(reason="This takes awhile to run. Only run when you want to test this functionality.")
@pytest.mark.usefixtures("xl_model_with_run")
class TestHCWAttendance(TestCase):
    def test_hours_worked_per_day(self):
        """
        Uses the sum of employee hours and contract hours from PBJ data as ground truth for hours worked per week per
        facility. This is the underlying target on which all of our healthcare worker assignment and attendance
        functions are built. Since some error is propagated at each step, we don't expect to hit these targets
        exactly, but we should be pretty close.
        """
        cms = calculate_target_worker_counts(read_staffing_data(), self.model.params, self.model.multiplier)
        cms["total_hrs"] = cms["emp_hrs"] + cms["ctr_hrs"]
        cms.set_index("federal_provider_number", inplace=True)
        cms["avg_hours_worked_per_week"] = np.nan

        nursing_homes = [facility for facility in self.model.nodes.facilities.values() if type(facility) == NursingHome]

        for site in nursing_homes:
            workers_per_day = [len(workers) for workers in site.worker_attendance.values()]
            avg_hours_worked_per_week = (
                np.nanmean(workers_per_day)
                * (self.model.params.full_time_hours_per_week / 5)
                * 7
                / self.model.multiplier
            )
            cms.loc[site.federal_provider_number, "avg_hours_worked_per_week"] = avg_hours_worked_per_week

        cms["ratio"] = cms.avg_hours_worked_per_week / cms.total_hrs

        # Mean ratio should be very close to 1 and standard deviation should not be more than 5%
        assert 0.99 < cms["ratio"].mean() < 1.01
        assert cms["ratio"].std() < 0.05

        def percent_ratio_deviant(ratios: pd.Series, deviance_threshold: float):
            ratio_deviance_from_1 = ratios - 1
            return sum([abs(d) > deviance_threshold for d in ratio_deviance_from_1]) / len(ratio_deviance_from_1)

        # No more than 25% of sites should have average hours worked more than 5% off from target
        assert percent_ratio_deviant(cms["ratio"], 0.05) < 0.25

        # No more than 5% of sites should have average hours worked more than 10% off from target
        assert percent_ratio_deviant(cms["ratio"], 0.1) < 0.05

        # No more than 1% of sites should have average hours worked more than 25% off from target
        assert percent_ratio_deviant(cms["ratio"], 0.25) < 0.01
