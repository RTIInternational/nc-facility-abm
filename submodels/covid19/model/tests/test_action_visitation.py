from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from model.state import LifeState
from submodels.covid19.model.state import COVIDState, VaccinationStatus


@pytest.mark.usefixtures("model")
class TestVisitation(TestCase):
    def test_assignment(self):
        self.model.nh_visitors = {}

        count = 5000
        for unique_id in self.model.unique_ids[0:count]:
            self.model.assign_nh_visitors(unique_id)

        # Get the IDS for each visitor type (1, 2, or 3) and count the number of each
        visitors = {1: [], 2: [], 3: []}
        counts = {k: 0 for k in range(0, 4)}
        for vd in self.model.nh_visitors.values():
            for i in range(1, len(vd) + 1):
                visitors[i].append(list(vd.keys())[i - 1])
            counts[len(vd)] += 1

        # Assert that the number of visitors matches the distribution
        vd = self.model.params.visitation_distribution
        target = vd["distribution"][0]
        value = counts[0] / count
        assert np.isclose(target, value, atol=0.02)
        target = vd["distribution"][1] - vd["distribution"][0]
        value = counts[1] / count
        assert np.isclose(target, value, atol=0.02)
        target = vd["distribution"][2] - vd["distribution"][1]
        value = counts[2] / count
        assert np.isclose(target, value, atol=0.02)
        target = vd["distribution"][3] - vd["distribution"][2]
        value = counts[3] / count
        assert np.isclose(target, value, atol=0.02)

        # Check the age distributions for each visitor (1st, 2nd, 3rd)
        for v_count in [1, 2, 3]:
            ages = self.model.age_group[visitors[v_count]]
            if len(ages) > 250:
                vc = pd.Series(self.model.age_group[visitors[v_count]]).value_counts(1).sort_index()
                values = vc.cumsum().values
                targets = self.model.params.visitor_age_distribution[v_count]

                for i in range(len(values)):
                    assert np.isclose(values[i], targets[i], atol=0.03)

    def test_visitation(self):
        # try assignment until we get 3 visitors
        unique_id = 0
        count = 0
        self.model.assign_nh_visitors(unique_id)
        while len(self.model.nh_visitors[unique_id]) != 3:
            self.model.nh_visitors.pop(unique_id)
            self.model.assign_nh_visitors(unique_id)
            count += 1
            if count > 100:
                raise ValueError("This test is flawed. Assignment of NH visitors is not producing 3 visitors..")

        unique_ids = list(self.model.nh_visitors[unique_id].keys())

        def make_visitations(
            iterations=100,
            location=0,
            life_status=LifeState.ALIVE,
            covid_status=COVIDState.SUSCEPTIBLE,
            vaccination_status=VaccinationStatus.NOTVACCINATED,
        ):
            self.model.movement.location[unique_ids] = location
            self.model.life[unique_ids] = life_status
            self.model.covid19[unique_ids] = covid_status
            self.model.vaccination_status[unique_ids] = vaccination_status

            self.model.nh_visits.data = []
            for _ in range(iterations):
                self.model.nh_visitation(unique_id)
            return self.model.nh_visits.make_events()

        # --------------------------------------------------------------------------------------------------------------
        # All at home, all alive, all susceptible, no vaccination required
        visits_5k = make_visitations(5000)
        # Assert each visitor visited according to their probability of visit
        vc = visits_5k.Visitor_ID.value_counts()
        for visitor_id in unique_ids:
            assert np.isclose(vc[visitor_id] / 5000, self.model.nh_visitors[unique_id][visitor_id], atol=0.03)

        # --------------------------------------------------------------------------------------------------------------
        # No one at home = no visitation
        visits = make_visitations(10, location=1)
        assert visits.shape[0] == 0

        # --------------------------------------------------------------------------------------------------------------
        # Visitors who have died to not visit
        visits = make_visitations(10, life_status=LifeState.DEAD)
        # Assert there are no visits
        assert visits.shape[0] == 0

        # --------------------------------------------------------------------------------------------------------------
        # Asymptomatic Symptoms Does not Reduce Visitation
        # Asymptomatic visitation rate should be the same as susceptible visitation rate
        visits_susceptible = make_visitations(5000, covid_status=COVIDState.SUSCEPTIBLE)
        visits_asymptomatic = make_visitations(5000, covid_status=COVIDState.ASYMPTOMATIC)
        assert np.isclose(visits_asymptomatic.shape[0] / 5000, visits_susceptible.shape[0] / 5000, atol=0.05)

        # --------------------------------------------------------------------------------------------------------------
        # Mild Symptoms should reduce visitation. Reduction should be based on parameter.
        visits_mild = make_visitations(5000, covid_status=COVIDState.MILD)
        v1 = visits_mild.shape[0] / 5000 / (1 - self.model.params.visitors_with_mild_who_stay_home)
        v2 = visits_susceptible.shape[0] / 5000
        assert np.isclose(v1, v2, atol=0.05)

        # --------------------------------------------------------------------------------------------------------------
        # Severe or Crtical Symptoms = No Visits
        visits_severe = make_visitations(100, covid_status=COVIDState.SEVERE)
        assert visits_severe.shape[0] == 0
        visits_critical = make_visitations(100, covid_status=COVIDState.CRITICAL)
        assert visits_critical.shape[0] == 0
