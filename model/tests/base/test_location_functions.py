from unittest.case import TestCase

import numpy as np
import pytest
import src.data_input as di
from model.facilities import Community, Hospital
from model.state import LocationCategories


@pytest.mark.usefixtures("model")
class TestLocationSetup(TestCase):
    def test_locations_class(self):

        hospitals = di.hospitals()

        # Test Community
        assert self.model.nodes.community == 0
        assert isinstance(self.model.nodes.facilities[0], Community)

        # Hospitals, NH, and LTCF All have the correct amount of facilities
        assert len(self.model.nodes.category_ints[LocationCategories.HOSPITAL.name]) == self.model.hospital_df.shape[0]
        assert len(self.model.nodes.category_ints[LocationCategories.HOSPITAL.name]) == hospitals.shape[0]
        assert di.nursing_homes().shape[0] == len(self.model.nodes.category_ints[LocationCategories.NH.name])
        assert di.ltachs().shape[0] == len(self.model.nodes.category_ints[LocationCategories.LT.name])

        # Categories must be one of the node categories
        for i, facility in self.model.nodes.facilities.items():
            assert facility.category in self.model.nodes.categories

        # Node Categories matches LocationCategories
        for i in self.model.nodes.categories:
            assert i in LocationCategories.__members__

        # add_facility works
        number_of_items = self.model.nodes.number_of_items
        hospital = Hospital(name="TEST FACILITY", ncdhsr_name="TEST NAME", county=1, n_beds=5, n_icu_beds=5)
        self.model.nodes.add_facility(hospital)
        assert self.model.nodes.number_of_items == number_of_items + 1

    def test_nh_ls_init(self):
        """Test that the nursing homes and LTACHS with at least 10 scaled beds are initialized to close to 70% full
        Test that most of the people in them are from nearby counties
        """
        probs = {LocationCategories.NH.name: di.facility_to_county_probabilities(LocationCategories.NH.name)}
        probs[LocationCategories.LT.name] = di.facility_to_county_probabilities(LocationCategories.LT.name)
        # get the information for each nursing home and LTACH
        f = (
            self.model.nodes.category_ints[LocationCategories.NH.name]
            + self.model.nodes.category_ints[LocationCategories.LT.name]
        )
        for i in f:
            facility = self.model.nodes.facilities[i]
            if facility.model_beds["total_beds"] > 10:
                capacity = facility.calculate_capacity()
                if facility.category == LocationCategories.NH.name:
                    target = facility.avg_capacity / facility.real_beds["total_beds"]
                else:
                    target = self.model.params.location.lt_fill_proportion
                assert np.isclose(capacity, target, atol=0.15)

                if facility.calculate_capacity(True) > 15:
                    counties = probs[facility.category][facility.name][1]
                    nearby = 0
                    far_away = 0
                    ac = []
                    for unique_id in self.model.nodes.facilities[i].agents:
                        if unique_id != -1:
                            ac.append(self.model.county_code[unique_id])
                            if self.model.county_code[unique_id] in counties[:50]:
                                nearby += 1
                            else:
                                far_away += 1
                    assert far_away < nearby

    def test_select_los(self):
        # Move to different locations, making sure the LOS is within reasonable values
        for _ in range(10):
            # LOS for Each category falls within the distribution
            for category in [LocationCategories.NH.name, LocationCategories.LT.name]:
                min_los = min(self.model.movement.los[category])
                max_los = max(self.model.movement.los[category])
                for f_int in self.model.nodes.category_ints[category]:
                    self.model.movement.current_los[0] = 0
                    self.model.movement.assign_los(unique_id=0, new_location=f_int)
                    assert min_los <= self.model.movement.current_los[0] <= max_los
            for hospital in self.model.nodes.category_ints[LocationCategories.HOSPITAL.name]:
                if hospital < 110:
                    min_los = min(self.model.movement.los[hospital])
                    max_los = max(self.model.movement.los[hospital])
                    self.model.movement.current_los[0] = 0
                    self.model.movement.assign_los(unique_id=0, new_location=hospital)
                    assert min_los <= self.model.movement.current_los[0] <= max_los

        # Providing non facility values should raise error
        for item in ["TEST_HOSPITAL", self.model.nodes.number_of_items + 1, np.nan, -1]:
            with pytest.raises(KeyError):
                self.model.movement.assign_los(0, item)

    def test_find_location_transitions(self):
        # All possible county, age, and location combinations have a potential destination
        for county in self.model.params.counties:
            for age in self.model.params.age_groups:
                for a_loc, _ in self.model.nodes.facilities.items():
                    if a_loc == self.model.nodes.community:
                        continue
                    assert self.model.movement.find_location_transitions(county, age, a_loc)[-1] > 0


@pytest.mark.usefixtures("model")
class TestMovement(TestCase):
    def test_community_movement_to_hospitals(self):
        """ Test if people leave the community correctly. Only test if population is high enough"""
        if self.model.params.num_agents >= 500_000:
            # Make first 1k people move: Set their probability to 1 and their current location to COMMUNITY
            count = 100
            self.model.movement.community_to_hospital_probabilities[0:count].fill(1)
            self.model.movement.location.values.fill(0)
            # No one should be in the community anymore
            self.model.movement.community_movement()
            for action in self.model.actions:
                action[0](**action[1])
            # Most people should move (if pop is limited, hospital bed counts may be to low to allow this)
            assert sum(self.model.movement.location.values[0:count] != 0) > (count * 0.9)

    def test_community_movement_to_nhs(self):
        pass

    def test_facility_movement(self):
        """ Test hospital movement"""
        self.model.time = 1
        self.model.movement.leave_facility_day = {}
        # Pick a big hospital
        hospital_ints = self.model.nodes.category_ints[LocationCategories.HOSPITAL.name]
        beds = [self.model.nodes.facilities[i].real_beds["total_beds"] for i in hospital_ints]
        h_int = hospital_ints[beds.index(max(beds))]

        # Find people at this facility
        unique_ids = self.model.unique_ids[self.model.movement.location.values == h_int]

        # Set them up to leave
        for unique_id in unique_ids:
            self.model.movement.current_los[unique_id] = 1
            self.model.movement.leave_facility_day[unique_id] = self.model.time

        # Run Function
        self.actions = []
        self.model.movement.facility_movement()
        for action in self.model.actions:
            action[0](**action[1])

        # Everyone should move
        new_locations = self.model.movement.location.values[unique_ids]
        assert all(new_locations != h_int)

        # Most people should go to the community
        if len(new_locations) > 10:
            assert sum(new_locations == 0) / len(new_locations) > 0.6


@pytest.mark.usefixtures("model_small")
class TestReadmission(TestCase):
    def test_readmission_movement(self):
        # Pick a big hospital
        hospital_ints = self.model.nodes.category_ints[LocationCategories.HOSPITAL.name]
        beds = [self.model.nodes.facilities[i].real_beds["total_beds"] for i in hospital_ints]
        h_int = hospital_ints[beds.index(max(beds))]

        # Clear the hospital
        agents = self.model.nodes.facilities[h_int].agents
        [self.model.nodes.facilities[h_int].remove_agent(i) for i in agents if i in self.model.unique_ids]

        self.model.time = 1
        unique_ids = self.model.unique_ids[0 : (sum(agents == -1) - 1)]
        for unique_id in unique_ids:
            self.model.movement.location.values[unique_id] = 0
            self.model.movement.readmission_date[unique_id] = self.model.time
            self.model.movement.readmission_location[unique_id] = h_int

        # Run the function
        self.model.movement.readmission_movement()
        for action in self.model.actions:
            action[0](**action[1])

        # Everyone should move to that new location
        assert all(self.model.movement.location[unique_ids] == h_int)

        # No one should be still in the readmission dictionaries
        self.model.time = self.model.time + 1
        self.model.movement.readmission_movement()
        assert all([i not in self.model.movement.readmission_location for i in unique_ids])
        assert all([i not in self.model.movement.readmission_date for i in unique_ids])


# TODO
# test_select_st_hospital
# update_location_transitions()
# select_location()
