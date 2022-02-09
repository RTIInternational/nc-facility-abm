from optabm.parameters import ParameterContainer
import src.data_input as di


class Parameters(ParameterContainer):
    def __init__(self):
        super().__init__()
        self.add_param("num_agents", di.nc_counties().Population.sum())
        self.add_param("time_horizon", 365)
        self.add_param("age_groups", [0, 1, 2])
        self.add_param("counties", list(range(1, 201, 2)))
        self.add_param("collect_daily_data", True)

        # Population
        self.add_param("nc_population", di.nc_counties().Population.sum())
        self.add_param("synpop_variables", ["County_Code", "Age_Group"])

        # Location
        self.add_param("location", LocationParameters())


class LocationParameters(ParameterContainer):
    def __init__(self):
        super().__init__()
        self.add_param("store_events", True)
        self.add_param("use_real_data", False)

        self.add_param("los_mean", 5, "Average length of stay for hospitals.")
        # Source: Phone call with CDC, we were told 93k and rounded to 100k.
        self.add_param("nh_death_proportion", 0.15, "Assume x% pf NH patients die each year.")
        # Source: CDC Email
        self.add_param("nh_to_community", 0.673, "Proportion of NH patients returning to the community")
        # Source: Toth_2017_CID_Potential Interventions LTACH Reduce Transmission CRE.pdf
        self.add_param("lt_to_hospital", 0.071, "Proportion of agents leaving LTACHs going to STACHs")
        self.add_param("lt_to_nh", 0.449, "Proportion of agents leaving LTACHs going to NHs")
        self.add_param("lt_death", 0.01, "Proportion of agents leaving LTACHs because of death")
        # Source: TODO: Estimate from RTI. We need lots of LTACH agents to be 65+ for NH reasons
        self.add_param("lt_65p", 0.75)
        self.add_param("nh_st_nh", 0.80, "x% of NH to STACH movement will return to a NH")

        # Fill Proportions
        self.add_param("lt_fill_proportion", 0.90, "Proportion of LTACHs that are full at start")
        self.add_param("acute_fill_proportion", 0.65, "How full are acute hospital beds?")
        self.add_param("icu_fill_proportion", 0.50, "How full are icu beds?")
        self.add_param("icu_reduction_multiplier", 0.73, "Reduce ICU probability by X%")

        # Readmission
        self.add_param("readission", 0.094, "Readmission rate of people readmitted within 30 days")
        self.add_param("readmission_days", 30, "Number of days for maximum readmission")

        # Distance measures
        self.add_param("NH_closest_n", 30)
        self.add_param("NH_attempt_count", 30, "The number of NHs to try before stoping.")
        self.add_param("LT_closest_n", 10)
        self.add_param("LT_attempt_count", 10, "The number of NHs to try before stoping.")
        self.add_param("max_distance", 200, "Maximum number of miles to try")

        # Length of Stay
        self.add_param("LT_LOS", {"distribution": "Gamma", "shape": 144, "support": 0.416667, "mean": 60, "std": 5})
        self.add_param(
            "HOSPITAL_LOS", {"distribution": "Gamma", "shape": 0.390625, "support": 12.8, "mean": 5, "std": 8}
        )


params = Parameters()
