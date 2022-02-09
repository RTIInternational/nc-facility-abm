from unittest import TestCase

import numpy as np
import pytest
from model.state import LocationCategories


@pytest.mark.usefixtures("model")
class TestLocationFiles(TestCase):
    def test_community(self):
        ct = self.model.community_transitions

        # 100 Counties, and 3 age groups
        assert ct.shape[0] == 300

        ct.reset_index(inplace=True)

        # Only 65+ For nursing home
        assert max(ct[ct.Age_Group < 2][LocationCategories.NH.name]) == 0
        assert min(ct[ct.Age_Group == 2][LocationCategories.NH.name]) > 0

    def test_county_discharge_files(self):
        discharges = self.model.discharges_df
        hospitals = self.model.hospital_df

        # There are 100 counties
        assert len(discharges) == 100

        # Make sure all hospitals are represented
        items = [name for name in hospitals.index if name in discharges.columns]
        assert len(items) == len(hospitals)

    def test_nh(self):
        temp_df = self.model.facility_transitions
        nh = temp_df[temp_df.Facility == LocationCategories.NH.name]

        # Anyone less than age group 2 should not have any probabilities (i.e. community == 1)
        assert nh[nh.Age_Group < 2].COMMUNITY.mean() == 1

        # Row should add to 1
        assert all(nh[[i.name for i in LocationCategories]].sum(axis=1) == 1)

        # Community, LT, and NH all have specific values
        nh = nh[nh.Age_Group == 2]
        assert nh[LocationCategories.COMMUNITY.name].mean() == self.model.params.location.nh_to_community
        assert round(nh[LocationCategories.HOSPITAL.name].mean(), 2) == round(
            (1 - self.model.params.location.nh_to_community), 2
        )
        assert nh[LocationCategories.LT.name].mean() == 0
        assert nh[LocationCategories.NH.name].mean() == 0

    def test_lt(self):
        temp_df = self.model.facility_transitions
        lt = temp_df[temp_df.Facility == LocationCategories.LT.name]

        # 3 rows for 3 age groups
        assert lt.shape[0] == len(self.model.params.age_groups)

        # Row should add to 1
        assert all(lt[[i.name for i in LocationCategories]].sum(axis=1) == 1)

        # LT can only be 0
        assert lt[LocationCategories.LT.name].mean() == 0

        # Only 65+ For nursing home
        assert max(lt[lt.Age_Group < 2][LocationCategories.NH.name]) == 0
        assert min(lt[lt.Age_Group == 2][LocationCategories.NH.name]) > 0

    def test_hospitals(self):
        temp_df = self.model.facility_transitions
        hospitals = self.model.hospital_df
        ht = temp_df[temp_df.Facility.isin(list(hospitals.index))]

        assert ht.shape[0] == hospitals.shape[0] * len(self.model.params.age_groups)

        # Rows should add to 1
        assert all(np.isclose(ht[[i.name for i in LocationCategories]].sum(axis=1), 1, atol=0.000001))

        # Only 65+ For nursing home
        assert max(ht[ht.Age_Group < 2][LocationCategories.NH.name]) == 0
        assert min(ht[ht.Age_Group == 2][LocationCategories.NH.name]) > 0

        # Only 50+ for LT
        assert max(ht[ht.Age_Group < 1][LocationCategories.LT.name]) == 0
        assert min(ht[ht.Age_Group > 0][LocationCategories.LT.name]) > 0
