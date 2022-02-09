from unittest import TestCase

import pytest
from submodels.covid19.model.state import COVIDState


@pytest.mark.usefixtures("model")
class TestRecovery(TestCase):
    def test_recovery(self):
        model = self.model

        # Simulate 100 COVID cases
        for i in range(0, 100):
            model.give_covid19(i)

        blocked_cases = model.blocked_covid_cases.make_events()

        # Every agent should have a recovery day or be in the blocked list
        unique_ids = []
        for i in range(0, 100):
            if i not in model.recovery_day:
                assert i in blocked_cases.Unique_ID.values
            else:
                unique_ids.append(i)

        model.time = model.params.infection_duration

        # Run Recovery
        model.action_recovery()
        # Perform all actions
        for action in model.actions:
            action[0](**action[1])

        # Assert that everyone has recovered (unless they are in a hospital)
        for i in unique_ids:
            if model.covid19.values[i] != COVIDState.RECOVERED:
                assert model.movement.location.values[i] != model.movement.community
