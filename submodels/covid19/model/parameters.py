from submodels.covid19.model.state import COVIDState

from model.parameters import Parameters


class CovidParameters(Parameters):
    """The following parameters are specific to the COVID-19 model only.

    All parameters are described and sourced (where applicable) in the COVID-19 ODD.
    """

    def __init__(self):
        super().__init__()
        self.add_param("start_date", "2021-12-15", "Start date for the simulation.")
        self.add_param("forecast_length", 30, "Number of days the simulation will run.")
        self.add_param("use_historical_case_counts", False, "Should historical case counts be used?")
        self.add_param("historical_cases_multiplier", 1, "A multiplier to increase/decrease historical cases")
        self.add_param("case_multiplier", 8, "The baseline underreporting value used in the SEIR model.")

        # Tracking Parameters
        self.add_param("track_hospitalizations", False, "Set True to track all hospitalizations.")
        self.add_param("simulate_nh_visitation", True, "Set True to simulate NH visitations")

        # SEIR Parameters
        self.add_param("r_effective", 1.5, "The effective reproduction number for COVID used in the SEIR model.")

        # Vaccination Rates
        self.add_param("community_vaccination_multiplier", 1, "Rate at which to increase community vaccinations.")
        self.add_param("nh_vaccination", 0.87, "Vaccination rate for NH residents.")
        self.add_param("healthcare_worker_vaccination", 0.80, "Vaccination rate for NH healthcare workers.")
        # Vaccine Effectiveness
        self.add_param("baseline_vaccine_effectiveness", 0.24, "Effectiveness of vaccination on preventing COVID.")
        self.add_param("new_vaccine_effectiveness", 0.24, "Effectiveness to use moving forward.")

        # Initial Hopistal Patients
        self.add_param(
            "initial_hospital_cases",
            {COVIDState.SEVERE.name: 1194, COVIDState.CRITICAL.name: 417},
            "Number of ICU and non-ICU hospitalizations to initiate the model with",
        )

        # COVID Age Distribution
        self.add_param("proportion_cases_post_vaccination", 0.39)
        self.add_param("covid_age_distribution", {"ages": [0, 50, 65, 105], "distribution": [0.70, 0.18, 0.12]})
        self.add_param("covid_hosp_age_distribution", {"ages": [0, 50, 65, 105], "distribution": [0.31, 0.25, 0.44]})

        # Case Severity
        self.add_param(
            "reported_notvaccinated_severity",
            {
                0: [0.05, 0.98536, 0.99703, 1.0],
                1: [0.05, 0.95408, 0.99068, 1.0],
                2: [0.05, 0.87878, 0.97539, 1.0],
            },
        )
        self.add_param(
            "reported_vaccinated_severity",
            {
                0: [0.25, 0.99093, 0.99903, 1.0],
                1: [0.25, 0.97157, 0.99699, 1.0],
                2: [0.25, 0.92494, 0.99204, 1.0],
            },
        )
        self.add_param(
            "nonreported_notvaccinated_severity",
            {
                0: [0.25, 1, 1, 1],
                1: [0.25, 1, 1, 1],
                2: [0.25, 1, 1, 1],
            },
        )
        self.add_param(
            "nonreported_vaccinated_severity",
            {
                0: [0.5, 1, 1, 1],
                1: [0.5, 1, 1, 1],
                2: [0.5, 1, 1, 1],
            },
        )
        self.add_param("infection_duration", 7)
        # LOS Parameters
        self.add_param("los_mean", 3)
        self.add_param("los_std", 5)
        self.add_param("los_min", 1)
        self.add_param("los_max", 50)

        # NH Visitation Parameters
        self.add_param("include_nh_visitation", True, "Boolean: Should NH visitation be simulated?")
        self.add_param("visitation_distribution", {"count": [0, 1, 2, 3], "distribution": [0.15, 0.60, 0.85, 1.00]})
        self.add_param(
            "visitor_age_distribution",
            {
                1: [0.1, 0.3, 1],
                2: [0.2, 0.6, 1],
                3: [0.4, 0.8, 1],
            },
        )
        self.add_param("visitor_frequency_distribution", {1: 15, 2: 5, 3: 1}, "Average visits per month")
        self.add_param("visitors_with_mild_who_stay_home", 0.6, "Probability someone with mild symptoms stays home")
        self.add_param("nh_visitiation_vaccine_required", False, "Is proof of vaccination required for NH visitors?")

        # Healthcare worker parameters
        self.add_param("include_hcw_attendance", True, "Boolean: Should healthcare worker attendance be simulated?")
        self.add_param(
            "workday_prob",
            5 / 7,
            "Probability that any given day is a working day for a full-time worker, expressed as the number of workdays out of the 7-day week",
        )
        self.add_param("full_time_hours_per_week", 39, "Hours worked per week by full-time healthcare workers")
        self.add_param(
            "pct_multi_site_workers",
            0.15,
            "Proportion of healthcare employees (not including contract workers) who work at 2 facilities",
        )
        self.add_param(
            "pct_time_second_site",
            1 / 3,
            "Proortion of time that workers who work at 2 facilities work at their secondary facility",
        )
        self.add_param(
            "pct_part_time_workers",
            0.15,
            "Proportion of healthcare employees (not including contract workers) who work part time",
        )
        self.add_param(
            "part_time",
            2 / 3,
            "Hours worked per week by part-time healthcare workers (as a fraction of full_time_hours_per_week)",
        )
        self.add_param("contract_worker_n_sites", 3, "Number of facilities at which each contract worker works")
        self.add_param("rapid_test_false_negative_rate_mild", 0.28)
        self.add_param("rapid_test_false_negative_rate_asymptomatic", 0.42)
