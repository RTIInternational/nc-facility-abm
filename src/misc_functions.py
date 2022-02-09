from functools import lru_cache

import pandas as pd
from optabm.misc_functions import create_cdf


@lru_cache()
def get_multiplier(params):
    return params.num_agents / params.nc_population


def get_inverted_distance_probabilities(county_to_facility_distances):
    """Given a dictionary of county to facility distances, invert it
    to get a probability distribution of each county given a facility based
    on distance
    Parameters
    ----------
    county_to_facility_distances: a dictionary of county IDs to lists of facilities
    and their distances to each county
    """
    facility_to_county_distances = {}
    for f in county_to_facility_distances["1"]:
        facility_to_county_distances[f["Name"]] = {}

    for county, distances in county_to_facility_distances.items():
        for f in distances:
            facility_to_county_distances[f["Name"]][int(county)] = f["distance_mi"]

    all_dist = pd.DataFrame(facility_to_county_distances)
    # we want closer distances to have higher probabilites so invert
    all_dist = 1 / all_dist
    # make these probability distributions
    prob_df = all_dist / all_dist.sum()

    prob_f_to_county_dict = {}
    for f in facility_to_county_distances.keys():
        # order by nearest county
        tmp_df = prob_df.sort_values(by=f, ascending=False)
        prob_f_to_county_dict[f] = [create_cdf(tmp_df[f].tolist()), tmp_df.index.tolist()]
    return prob_f_to_county_dict
