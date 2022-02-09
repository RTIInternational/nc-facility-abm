from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from model.state import LocationCategories
import src.data_input as di


@pytest.mark.usefixtures("model")
class TestInitiation(TestCase):
    def test_location_initiation(self):
        """ Make sure the number of agents matches the requested amount"""
        facilities = self.model.nodes.facilities
        location = pd.Series([facilities[i].category for i in self.model.movement.location.values])
        vc = location.value_counts()

        # ----- Nursing Homes: Within 2% given the large amount of beds
        nh = di.nursing_homes()
        nh_beds = sum([facilities[i].model_beds["total_beds"] for i in self.model.nodes.category_ints["NH"]])
        target = nh.average_number_of_residents_per_day.sum() / nh.Beds.sum()
        assert np.isclose(vc.loc[LocationCategories.NH.name] / nh_beds, target, atol=0.03)

        # ----- LTACHs: Within 10% given the small amount of beds
        lt_beds = sum([facilities[i].model_beds["total_beds"] for i in self.model.nodes.category_ints["LT"]])
        target = self.model.params.location.lt_fill_proportion
        assert np.isclose(vc.loc[LocationCategories.LT.name] / lt_beds, target, atol=0.1)

        # ----- Hospitals: Each hospital should be within 5%
        for facility in self.model.nodes.category_ints["HOSPITAL"]:
            facility = self.model.nodes.facilities[facility]
            if facility.model_beds["total_beds"] < 5:
                pass
            real_facility = self.model.hospital_df.loc[facility.name]
            real_capacity = real_facility[["Acute_Agents", "ICU_Agents"]].sum() / real_facility.Beds
            model_capacity = facility.calculate_capacity("all")
            assert np.isclose(real_capacity, model_capacity, atol=0.5)
