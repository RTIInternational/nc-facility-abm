import itertools
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import pylab as plt
from model.north_carolina_nodes import NcNodeCollection
from numba import njit
from submodels.covid19.experiments.network_analysis.src.make_scenario import experiment_dir
from submodels.covid19.model.parameters import CovidParameters
from submodels.covid19.model.state import COVIDState
from tqdm import tqdm

import src.data_input as di
from src.misc_functions import get_multiplier

AUTO_OPEN = False


@njit
def rolling_shifts(locations: np.array, times: np.array, n: int = 8):
    """Count the shifts worked at a facility other than the original facility
    NOTE: n should be days + 1. For a 7 day window, use n=8.
    ASSUMPTION: The locations are part of a dataframe filtered by ["Unique_ID", "Time"]
    We look at the time to determine if a new ID is reached. Time must be less than Time(i) + 7, but also greater than
    or equal to Time(i)
    """
    counts = np.zeros(len(locations))
    for i in range(0, len(locations)):
        possible_times = times[i : (i + n)]
        f1 = (times[i] <= possible_times) & (possible_times < times[i] + n)
        new_facility = locations[i : (i + n)] != locations[i]
        counts[i] = (f1 & new_facility).sum()
    return counts


@njit
def rolling_unique_facilities(locations: np.array, times: np.array, n: int = 8):
    """Count the number of unique facilities within a window.
    n should be days + 1. For a 7 day window, use n=8.
    ASSUMPTION: The locations are part of a dataframe filtered by ["Unique_ID", "Time"]
    We look at the time to determine if a new ID is reached. Time must be less than Time(i) + 7, but also greater than
    or equal to Time(i)
    """
    counts = np.zeros(len(locations))
    for i in range(0, len(locations)):
        possible_times = times[i : (i + n)]
        f1 = (times[i] <= possible_times) & (possible_times < times[i] + n)
        counts[i] = len(np.unique(locations[i : (i + n)][f1]))
    return counts


@njit
def rolling_additional_facilities(locations: np.array, times: np.array, n: int = 8):
    """Count the number of additional facilities visitited within a window.
    n should be days + 1. For a 7 day window, use n=8.
    ASSUMPTION: The locations are part of a dataframe filtered by ["Unique_ID", "Time"]
    We look at the time to determine if a new ID is reached. Time must be less than Time(i) + 7, but also greater than
    or equal to Time(i)
    """
    counts = np.zeros(len(locations))
    for i in range(0, len(locations)):
        possible_times = times[i : (i + n)]
        f1 = (times[i] <= possible_times) & (possible_times < times[i] + n)
        counts[i] = (locations[i : (i + n)][f1] != locations[i]).sum()
    return counts


def create_hcw_df(output_dir: Path):
    """Open and update the HCW dataframe to get"""
    hcw_df = pd.read_csv(output_dir.joinpath("hcw_attendance.csv"))

    # ----- Was there an asymptomatic HCW, or Visitor at that location on that day?
    visitor_df = pd.read_csv(output_dir.joinpath("nh_visits.csv"))
    # Number of Asymptomatic & Mild Visitors Per Facility Per Day
    visitor_df = visitor_df[visitor_df.COVID_State.isin([COVIDState.ASYMPTOMATIC, COVIDState.MILD])]
    visitor_dict = visitor_df.groupby(["Time", "Location", "COVID_State"]).size().to_dict()
    # Number of Asymptomatic & Mild HCWs Per Facility Per Day
    temp_hcw_df = hcw_df[hcw_df.COVID_State.isin([COVIDState.ASYMPTOMATIC, COVIDState.MILD])]
    hcw_dict = temp_hcw_df.groupby(["Time", "Location", "COVID_State"]).size().to_dict()
    # Did they come into contact with an asymptomatic or mild HCW or visitor on the same day?
    for state in [COVIDState.ASYMPTOMATIC, COVIDState.MILD]:
        col = [f"{ state.name.title()}_HCW"]
        hcw_df[col] = [i in hcw_dict.keys() for i in zip(hcw_df.Time, hcw_df.Location, [state] * len(hcw_df))]
        col = [f"{ state.name.title()}_Visitor"]
        hcw_df[col] = [i in visitor_dict.keys() for i in zip(hcw_df.Time, hcw_df.Location, [state] * len(hcw_df))]

    # ----- Count the number of shifts worked at a different facility within X days
    hcw_df = hcw_df.sort_values(["Unique_ID", "Time"])
    locations = hcw_df.Location.values
    times = hcw_df.Time.values
    # Number of shifts worked at a different facility within X days
    hcw_df["Shifts_3d"] = rolling_shifts(locations, times, 3 + 1)
    hcw_df["Shifts_7d"] = rolling_shifts(locations, times, 7 + 1)
    # hcw_df["Additional_Facilities_7d"] = rolling_unique_facilities(locations, times, 7 + 1)
    # hcw_df["Additional_Facilities_3d"] = rolling_unique_facilities(locations, times, 3 + 1)
    # hcw_df["Days_Worked_Other_Facilities_7d"] = rolling_additional_facilities(locations, times, 7 + 1)
    # hcw_df["Days_Worked_Other_Facilities_3d"] = rolling_additional_facilities(locations, times, 3 + 1)
    return hcw_df


def create_g_graph(edge_df):
    """Create the initial network graph"""
    g = nx.Graph()
    g.add_weighted_edges_from(edge_df[["L1", "L2", "Weight"]].values)
    nursing_homes = di.nursing_homes()
    attrs = {}
    nodes = NcNodeCollection(1)
    for node in g.nodes:
        nh_id = nodes.facilities[node].federal_provider_number
        row = nursing_homes[nursing_homes.federal_provider_number == nh_id]
        attrs[node] = {"pos": (row.lon.values[0], row.lat.values[0])}
    nx.set_node_attributes(g, attrs)
    return g


def create_degree_histogram(output_dir: Path, g: nx.classes.graph.Graph):
    """Create a histogram of degree counts (the number of facilities connected to that facility through staff)"""
    degree_counts = pd.DataFrame([g.degree(i) for i in g.nodes], columns=["Count"])
    fig = px.histogram(degree_counts, x="Count", color_discrete_sequence=["rgba(27,158,119,.50)"])
    fig.update_layout(
        title="Number of Facilities A Nursing Home Shares Staff Members With",
        xaxis_title="Number of Unique Facilities",
        yaxis_title="Number of Nursing Homes",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(showline=True, linewidth=0.5, gridcolor="rgba(27,158,119,.25)")
    plotly.offline.plot(fig, filename=str(output_dir.joinpath("hcw_degree_histogram.html")), auto_open=AUTO_OPEN)


def create_connections_histogram(output_dir: Path, edge_df: pd.DataFrame):
    fig = px.histogram(edge_df, x="Connections", color_discrete_sequence=["rgba(27,158,119,.50)"])
    fig.update_layout(
        title="Number of Staff Members Connecting 2 Nursing Homes",
        xaxis_title="Number of Staff Members",
        yaxis_title="Number of Connections (2 Nursing Homes)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(showline=True, linewidth=0.5, gridcolor="rgba(27,158,119,.25)")
    plotly.offline.plot(fig, filename=str(output_dir.joinpath("hcw_degree_histogram.html")), auto_open=AUTO_OPEN)


def draw_base_network(output_dir: Path, g: nx.classes.graph.Graph):
    pos = nx.get_node_attributes(g, "pos")
    weights = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_nodes(g, pos, node_size=20, node_color="blue", alpha=0.7)
    nx.draw_networkx_edges(
        g, pos, edgelist=weights.keys(), width=[i * 5 for i in weights.values()], edge_color="lightblue", alpha=0.9
    )
    plt.savefig(output_dir.joinpath("base_nh_network.png"), format="PNG")


def create_edge_df(hcw_df):
    """Create an edge df where the edges represent connections between two facilities.
    Two facilities are connected if they have two workers that work at both facilities. The weight is determined
    by the count of workers that work at both locations."""
    gb = hcw_df.groupby(["Unique_ID"]).Location
    edges = []
    for group in tqdm(gb):
        locations = group[1].unique()
        if len(locations) > 0:
            locations.sort()
            for edge in itertools.combinations(locations, 2):
                edges.append(edge)
    edge_df = pd.DataFrame(edges, columns=["L1", "L2"])
    edge_df = edge_df.groupby(["L1", "L2"]).size().reset_index()
    edge_df.rename(columns={0: "Weight"}, inplace=True)
    edge_df["Connections"] = edge_df["Weight"]
    edge_df["Weight"] = edge_df["Weight"] / edge_df["Weight"].max()
    return edge_df


# ----- Read Data
def network_analysis(output_dir: Path):
    # ----- Create the HCW DF
    hcw_df = create_hcw_df(output_dir)
    hcw_df.to_csv(output_dir.joinpath("hcw_df.csv"))

    # ----- Create the edge df
    edge_df = create_edge_df(hcw_df)

    # ----- Create histograms
    g = create_g_graph(edge_df)
    # draw_base_network(output_dir, g)
    create_degree_histogram(output_dir, g)
    create_connections_histogram(output_dir, edge_df)

    # ----- Create Growth Visualization
    create_network_visualization(hcw_df, output_dir)
    return hcw_df


def create_network_visualization(hcw_df, output_dir):
    # Look at the network at 1 day, 2 days, ... 7 days.
    location = hcw_df.Location.value_counts().index[0]
    time = 10
    original_staff_members = hcw_df[(hcw_df.Location == location) & (hcw_df.Time == time)].Unique_ID.values
    staff = original_staff_members.copy()
    fig, axes = plt.subplots(nrows=2, ncols=4)
    for i in range(1, 9):
        plt.subplot(2, 4, i)
        day = time + i
        # Where did those staff members work the next day?
        temp_hcw_df = hcw_df[hcw_df.Unique_ID.isin(staff) & (hcw_df.Time == (day))]
        # Find all staff that worked at one of these locations
        temp_hcw_df = hcw_df[
            hcw_df.Location.isin(temp_hcw_df.Location.unique()) & (hcw_df.Time <= day) & (hcw_df.Time >= time)
        ]
        temp_edge_df = create_edge_df(temp_hcw_df)
        if temp_edge_df.shape[0] == 0:
            temp_edge_df.loc[0] = [location, location, 0, 0]
        temp_g = create_g_graph(temp_edge_df)
        weights = nx.get_edge_attributes(temp_g, "weight")
        # pos = nx.spring_layout(temp_g)
        pos = nx.get_node_attributes(temp_g, "pos")
        colors = []
        for item in pos.keys():
            if item == location:
                colors.append("red")
            else:
                colors.append("blue")
        nx.draw_networkx_nodes(temp_g, pos, node_size=20, node_color=colors, alpha=0.7)
        nx.draw_networkx_edges(
            temp_g,
            pos,
            edgelist=weights.keys(),
            width=[i * 3 for i in weights.values()],
            edge_color="lightblue",
            alpha=0.9,
        )
        # Update the staff list
        staff = np.union1d(staff, temp_hcw_df.Unique_ID.unique())
    plt.savefig(output_dir.joinpath("network_growth_visualization.png"), format="PNG")


def main():
    # ----- Setup the base experiment
    params = CovidParameters()
    params.update_from_file("submodels/covid19/experiments/network_analysis/scenario_base_small/parameters.yml")
    multiplier = 1 / get_multiplier(params)
    # ----- Run network analysis on each scenario
    hcw_dfs = {}
    for scenario_dir in experiment_dir.glob("scenario*"):
        if "scenario_base" in scenario_dir.name:
            continue
        for run in scenario_dir.glob("run_*"):
            output_dir = run.joinpath("model_output")
            hcw_dfs[scenario_dir.name] = network_analysis(output_dir)

    out_dir = Path("submodels/covid19/experiments/network_analysis/analysis")
    # Aggregate that analysis
    hcw_result = []
    for scenario, hcw_df in hcw_dfs.items():
        v1 = int(hcw_df["Shifts_3d"].sum() * multiplier)
        v2 = int(hcw_df["Shifts_7d"].sum() * multiplier)
        hcw_result.append([scenario, v1, v2])
    hcw_summary_df = pd.DataFrame(hcw_result, columns=["Scenario", "Shifts Within 3 Days", "Shifts Within 7 Days"])
    hcw_summary_df.sort_values(by=["Scenario"], inplace=True)
    hcw_summary_df.to_csv(out_dir.joinpath("hcw_summary.csv"))


def todo():
    pass
    # centrality_metrics = pd.DataFrame(nx.closeness_centrality(g).items(), columns=["Node", "Closeness_Centrality"])
    # centrality_metrics["Degree_Centrality"] = [i[1] for i in nx.degree_centrality(g).items()]
    # centrality_metrics.mean


if __name__ == "__main__":
    main()
