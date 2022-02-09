import json
from pathlib import Path

import src.data_input as di
import yaml
from geopy import distance
from model.state import LocationCategories


filepaths = yaml.load(Path("config/filepaths.yaml").read_text(), Loader=yaml.loader.SafeLoader)


if __name__ == "__main__":

    county_centroids = json.loads(Path("data/geography/county_centroids.json").read_text())

    county_ids = di.nc_counties()[["County", "County_Code"]].assign(County=lambda df: df["County"].str.upper())
    county_id_map = {row.County: row.County_Code for row in county_ids.itertuples()}

    locations = {
        LocationCategories.NH.name: di.nursing_homes(),
        LocationCategories.LT.name: di.ltachs(),
        LocationCategories.HOSPITAL.name: di.hospitals(),
    }
    for key, df in locations.items():

        facility_locations = df.rename(columns={"LAT": "lat", "LON": "lon"}).dropna(subset=["lat", "lon"])
        facility_data = [(row["Name"], row.lat, row.lon) for inde, row in facility_locations.iterrows()]

        centroid_distances = {}

        for county, data in county_centroids.items():
            county_id = county_id_map[county]
            county_lat_lon = (data[0], data[1])
            county_distances = []
            for hospital, lat, lon in facility_data:
                hospital_lat_lon = (lat, lon)
                dist = distance.distance(county_lat_lon, hospital_lat_lon).miles
                county_distances.append({"Name": hospital, "distance_mi": dist})
            centroid_distances[county_id] = county_distances

        for county, distances in centroid_distances.items():
            centroid_distances[county] = list(sorted(distances, key=lambda x: x["distance_mi"]))

        location = Path(filepaths["geography_folder"]["path"], f"county_{key}_distances_sorted.json")
        location.write_text(json.dumps(centroid_distances, indent=4))
