from unittest import TestCase

import pytest
import src.data_input as di
from model.state import LocationCategories


@pytest.mark.usefixtures("model_small")
class TestNodes(TestCase):

    # ----- Test Counts
    def test_community_count(self):
        """1 Community Node"""
        facilities = self.model.nodes.facilities
        nodes = [facilities[f].category == LocationCategories.COMMUNITY.name for f in facilities]
        assert sum(nodes) == 1

    def test_hospital_count(self):
        """Hospital Nodes Equal to Input File"""
        facilities = self.model.nodes.facilities
        nodes = [facilities[f].category == LocationCategories.HOSPITAL.name for f in facilities]
        assert sum(nodes) == self.model.hospital_df.shape[0]

    def test_lt_count(self):
        """LTACHs"""
        facilities = self.model.nodes.facilities
        nodes = [facilities[f].category == LocationCategories.LT.name for f in facilities]
        assert sum(nodes) == di.ltachs().shape[0]

    def test_nh_count(self):
        """Nursing Homes"""
        facilities = self.model.nodes.facilities
        nodes = [facilities[f].category == LocationCategories.NH.name for f in facilities]
        assert sum(nodes) == di.nursing_homes().shape[0]

    # ----- Test Attributes
    def test_facility_attrs(self):
        """Test existence of attributes in table"""
        hospital_ints = self.model.nodes.category_ints[LocationCategories.HOSPITAL.name]
        attrs = ["name", "category", "county", "real_beds", "model_beds", "agents"]
        for hospital_int in hospital_ints:
            for attr in attrs:
                assert hasattr(self.model.nodes.facilities[hospital_int], attr)

    def test_facility_ids(self):
        nodes_int_ids = [i for i, n in self.model.nodes.facilities.items()]
        assert min((n for n in nodes_int_ids)) == 0
        assert max((n for n in nodes_int_ids)) == len(self.model.nodes.facilities) - 1

    # ----- Test Tests
    def test_location_state(self):
        assert hasattr(self.model, "movement")  # movement is a NorthCarolina object
        assert len(self.model.movement.location.values) == len(self.model.population)

    def test_life_state(self):
        assert hasattr(self.model, "life")
        assert len(self.model.life.values) == len(self.model.population)

    # ----- Test Demographics
    def test_columns(self):
        columns = self.model.params.synpop_variables
        for column in columns:
            assert hasattr(self.model, column.lower())
            assert len(getattr(self.model, column.lower())) == len(self.model.population)

    def test_concurrent_conditions(self):
        assert hasattr(self.model, "concurrent_conditions")
        assert len(self.model.concurrent_conditions) == len(self.model.population)

    # ----- Test Temporary States
    def test_los(self):
        assert hasattr(self.model.movement, "current_los")
        self.model.step()
        # bool({}) returns False, this is asserting the dictionary is not empty
        assert bool(self.model.movement.current_los), "current_los empty after 1 step"

    def test_leave_facility_day(self):
        assert hasattr(self.model.movement, "leave_facility_day")
        self.model.step()
        assert bool(self.model.movement.leave_facility_day), "leave_facility_day empty after 1 step"

    def test_previous_location(self):
        assert hasattr(self.model.movement, "leave_facility_day")
        self.model.step()
        assert len(set(self.model.movement.location.previous)) > 1, "location.previous unchanged after 1 step"

    def test_readmission_dictionaries(self):
        assert hasattr(self.model.movement, "readmission_date")
        assert hasattr(self.model.movement, "readmission_location")
        for day in range(10):
            self.model.time = day
            self.model.step()  # takes time for readmission
        assert bool(self.model.movement.readmission_date), "readmission_date empty after 10 steps"
        assert bool(self.model.movement.readmission_location), "readmission_location empty after 10 steps"
