from typing import Dict, Union

import src.data_input as di

from model.facilities import Community, Hospital, LongTermCareFacility, NursingHome
from model.state import LocationCategories


class NcNodeCollection:
    """A class for holding all location nodes for North Carolina"""

    def __init__(self, multiplier: float = 1):

        self.community = 0
        self.number_of_items = 0
        self.facilities: Dict[int, Union[Community, Hospital, NursingHome, LongTermCareFacility]] = dict()
        self.category_ints: Dict[int, str] = dict()
        self.name_to_int: Dict[str, int] = dict()
        self.categories = [i.name for i in LocationCategories]

        # Add Community
        self.add_facility(facility=Community())
        # Add Hospitals
        hospitals = di.hospitals()
        for _, row in hospitals.iterrows():
            hospital = Hospital(
                name=row.Name, ncdhsr_name=row.Facility, county=row.County, n_beds=row.Beds, n_icu_beds=row["ICU Beds"]
            )
            hospital.apply_bed_multiplier(multiplier)
            self.add_facility(hospital)
        # Add Nursing Homes
        nhs = di.nursing_homes()
        for row in nhs.itertuples():
            nh = NursingHome(
                name=row.Name,
                federal_provider_number=row.federal_provider_number,
                beds=int(row.Beds),
                avg_capacity=row.average_number_of_residents_per_day,
                model_int=self.number_of_items,
                county_code=row.County_Code,
            )
            nh.apply_bed_multiplier(multiplier)
            self.add_facility(nh)
        # Add LTACHs
        lt_ids = di.ltachs()
        for row in lt_ids.itertuples():
            ltach = LongTermCareFacility(beds=row.Beds, name=row.Name, model_int=self.number_of_items)
            ltach.apply_bed_multiplier(multiplier)
            self.add_facility(ltach)

    def __repr__(self):
        repr_values = [
            "NcNodeCollection",
            f"Hospitals: {len(self.category_ints[LocationCategories.HOSPITAL.name])}",
            f"Nursing Home: {len(self.category_ints[LocationCategories.NH.name])}",
            f"LTACHs: {len(self.category_ints[LocationCategories.LT.name])}",
        ]
        return " - ".join(repr_values)

    def n_per_category(self, category: str) -> int:
        return len(self.category_ints[category])

    def add_facility(self, facility):
        category = facility.category
        facility.model_int = self.number_of_items
        self.facilities[facility.model_int] = facility
        self.category_ints[category] = self.category_ints.get(category, []) + [facility.model_int]
        self.name_to_int[facility.name] = facility.model_int
        self.number_of_items += 1
