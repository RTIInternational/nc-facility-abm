from copy import deepcopy
from itertools import combinations
from unittest import TestCase

import pytest
from model.facilities import NursingHome
from submodels.covid19.src.healthcare_workers import (
    calculate_target_worker_counts,
    read_county_distance_data,
    read_staffing_data,
)


@pytest.mark.usefixtures("model")
class TestHealthcareWorkers(TestCase):
    def test_worker_creation(self):
        # No agent should be included in multiple worker lists
        worker_lists = [
            self.model.healthcare_workers["single_site_full_time"]["workers"],
            self.model.healthcare_workers["single_site_part_time"]["workers"],
            self.model.healthcare_workers["multi_site"]["workers"],
            self.model.healthcare_workers["contract"]["workers"],
        ]
        for combo in combinations(worker_lists, 2):
            assert set(combo[0]).isdisjoint(combo[1])

    def test_single_site_worker_assignment(self):
        nursing_homes = [facility for facility in self.model.nodes.facilities.values() if type(facility) == NursingHome]
        worker_counts = calculate_target_worker_counts(read_staffing_data(), self.model.params, self.model.multiplier)
        # Each single site worker should have exactly one site
        for worker in self.model.healthcare_workers["single_site_full_time"]["workers"]:
            assert len([site for site in nursing_homes if worker in site.single_site_full_time_workers]) == 1

        for worker in self.model.healthcare_workers["single_site_part_time"]["workers"]:
            assert len([site for site in nursing_homes if worker in site.single_site_part_time_workers]) == 1

        for site in nursing_homes:
            # Each site should have the right number of single site full time and part time workers
            row = worker_counts[worker_counts.federal_provider_number.eq(site.federal_provider_number)]
            target_single_site_full_time_workers = int(row.target_single_site_full_time_workers)
            target_single_site_part_time_workers = int(row.target_single_site_part_time_workers)

            assert len(site.single_site_full_time_workers) == target_single_site_full_time_workers
            assert len(site.single_site_part_time_workers) == target_single_site_part_time_workers

            # Ensure no duplicates in worker lists
            assert len(site.single_site_full_time_workers) == len(set(site.single_site_full_time_workers))
            assert len(site.single_site_part_time_workers) == len(set(site.single_site_part_time_workers))

    def test_multi_site_worker_assignment(self):
        nursing_homes = [facility for facility in self.model.nodes.facilities.values() if type(facility) == NursingHome]
        worker_counts = calculate_target_worker_counts(read_staffing_data(), self.model.params, self.model.multiplier)
        # Each multi site worker should have exactly one primary site and exactly one secondary site, which should not
        # be the same.
        for worker in self.model.healthcare_workers["multi_site"]["workers"]:
            primary_sites = [site for site in nursing_homes if worker in site.multi_site_primary_workers]
            secondary_sites = [site for site in nursing_homes if worker in site.multi_site_secondary_workers]

            assert len(primary_sites) == 1
            assert len(secondary_sites) == 1
            assert primary_sites[0] != secondary_sites[0]

        for site in nursing_homes:
            # Each site should have the right number of primary and secondary multi site workers, none of which should
            # be the same.
            row = worker_counts[worker_counts.federal_provider_number.eq(site.federal_provider_number)]
            target_multi_site_primary_workers = int(row.target_multi_site_primary_workers)
            target_multi_site_secondary_workers = int(row.target_multi_site_secondary_workers)

            assert len(site.multi_site_primary_workers) == target_multi_site_primary_workers
            assert len(site.multi_site_secondary_workers) == target_multi_site_secondary_workers
            assert set(site.multi_site_primary_workers).isdisjoint(site.multi_site_secondary_workers)

            # Ensure no duplicates in worker lists
            assert len(site.multi_site_primary_workers) == len(set(site.multi_site_primary_workers))
            assert len(site.multi_site_secondary_workers) == len(set(site.multi_site_secondary_workers))

    def test_contract_worker_assignment(self):
        nursing_homes = [facility for facility in self.model.nodes.facilities.values() if type(facility) == NursingHome]
        worker_counts = calculate_target_worker_counts(read_staffing_data(), self.model.params)
        # Each contract worker should have exactly three sites, none of which should be the same.
        for worker in self.model.healthcare_workers["contract"]["workers"]:
            sites = [site for site in nursing_homes if worker in site.contract_workers]

            assert len(sites) == 3
            assert len(sites) == len(set(sites))

        for site in nursing_homes:
            # Each site should have the right number of contract workers
            row = worker_counts[worker_counts.federal_provider_number.eq(site.federal_provider_number)]
            target_contract_workers = int(row.target_contract_workers)
            assert len(site.contract_workers) == target_contract_workers

            # Ensure no duplicates
            assert len(site.contract_workers) == len(set(site.contract_workers))

    def test_worker_exclusivity(self):
        nursing_homes = [facility for facility in self.model.nodes.facilities.values() if type(facility) == NursingHome]
        # No worker should be assigned to the same site in multiple categories
        for site in nursing_homes:
            worker_lists = [
                site.single_site_full_time_workers,
                site.single_site_part_time_workers,
                site.multi_site_primary_workers,
                site.multi_site_secondary_workers,
                site.contract_workers,
            ]
            for combo in combinations(worker_lists, 2):
                assert set(combo[0]).isdisjoint(combo[1])

    def test_worker_location(self):
        nursing_homes = [facility for facility in self.model.nodes.facilities.values() if type(facility) == NursingHome]
        distances = read_county_distance_data()
        # Single site workers should work and live in the same county. Multi site workers' primary site should be in
        # the same county where they live.
        for worker in self.model.healthcare_workers["single_site_full_time"]["workers"]:
            county_of_residence = int(self.model.population.iloc[worker].County_Code)
            site_county = [site for site in nursing_homes if worker in site.single_site_full_time_workers][
                0
            ].county_code
            assert county_of_residence == site_county

        for worker in self.model.healthcare_workers["single_site_part_time"]["workers"]:
            county_of_residence = int(self.model.population.iloc[worker].County_Code)
            site_county = [site for site in nursing_homes if worker in site.single_site_part_time_workers][
                0
            ].county_code
            assert county_of_residence == site_county

        for worker in self.model.healthcare_workers["multi_site"]["workers"]:
            county_of_residence = int(self.model.population.iloc[worker].County_Code)
            site_county = [site for site in nursing_homes if worker in site.multi_site_primary_workers][0].county_code
            assert county_of_residence == site_county

        # No more than 10% of multi-site workers should have a secondary site in a county more than 50 miles from their
        # home county.
        secondary_site_distances = []

        for worker in self.model.healthcare_workers["multi_site"]["workers"]:
            county_of_residence = int(self.model.population.iloc[worker].County_Code)
            site_county = [site for site in nursing_homes if worker in site.multi_site_secondary_workers][0].county_code
            if county_of_residence == site_county:
                secondary_site_distances.append(0)
            else:
                distance = float(
                    distances[
                        distances.county_from.eq(county_of_residence) & distances.county_to.eq(site_county)
                    ].distance
                )
                secondary_site_distances.append(distance)

        def percent_distances_over(distances: list, critical_value: int):
            return sum([d > critical_value for d in distances]) / len(distances)

        assert percent_distances_over(secondary_site_distances, 50) < 0.1

        # No more than 5% of multi-site workers should have a secondary site in a county more than 100 miles from their
        # home county.
        assert percent_distances_over(secondary_site_distances, 100) < 0.05

        # No more than 40% of contract workers should have a site in a county more than 50 miles from their home county.
        contract_site_distances = []

        for worker in self.model.healthcare_workers["contract"]["workers"]:
            county_of_residence = int(self.model.population.iloc[worker].County_Code)
            site_counties = [site.county_code for site in nursing_homes if worker in site.contract_workers]
            site_distances = []
            for site_county in site_counties:
                if county_of_residence == site_county:
                    site_distances.append(0)
                else:
                    site_distances.append(
                        float(
                            distances[
                                distances.county_from.eq(county_of_residence) & distances.county_to.eq(site_county)
                            ].distance
                        )
                    )
            contract_site_distances.append(max(site_distances))

        assert percent_distances_over(contract_site_distances, 50) < 0.4

        # No more than 20% of contract workers should have a site in a county more than 100 miles from their home county.
        assert percent_distances_over(contract_site_distances, 100) < 0.2

        # No more than 5% of contract workers should have a site in a county more than 200 miles from their home county.
        assert percent_distances_over(contract_site_distances, 200) < 0.05

    def test_contract_multiplier(self):
        worker_counts = calculate_target_worker_counts(read_staffing_data(), self.model.params, self.model.multiplier)

        unadjusted_params = deepcopy(self.model.params)
        setattr(unadjusted_params, "contract_hours_multiplier", 1)

        unadjusted_worker_counts = calculate_target_worker_counts(
            read_staffing_data(), unadjusted_params, self.model.multiplier
        )

        for i, row in worker_counts.iterrows():
            assert row["total_hrs"] == row["emp_hrs"] + row["ctr_hrs"]
            assert row["ctr_hrs"] <= row["total_hrs"]
            if row["ctr_hrs"] < row["total_hrs"]:
                assert (
                    row["ctr_hrs"]
                    == unadjusted_worker_counts["ctr_hrs"][i] * self.model.params.contract_hours_multiplier
                )
