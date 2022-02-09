from copy import copy
from unittest import TestCase

import pytest
from model.state import LifeState, LocationCategories


@pytest.mark.usefixtures("model")
class TestSteps(TestCase):

    # ----- Life Step
    def test_life_step(self):
        """For agents at each location, send the agent to the community to open up the bed that they were occupying."""
        for location_int in [self.model.nodes.category_ints[i.name][0] for i in LocationCategories]:
            # create
            agent_locations = self.model.movement.location.values
            community_agent_id = next(id_ for id_, location in enumerate(agent_locations) if location != location_int)
            # act
            self.model.life.death(community_agent_id)
            # assert
            assert self.model.life.values[community_agent_id] == LifeState.DEAD.value, "Agent state not DEAD"
            new_agent_locations = self.model.movement.location.values
            assert new_agent_locations[community_agent_id] == 0, "Agent not added to community"

    # ----- Location Step
    def test_location_step(self):
        "* Administer a location update (outlined in Section 7. Submodels) for any agent whose LOS ends on the current day"
        self.model.step()
        agents_leaving = [k for k, v in self.model.movement.leave_facility_day.items() if v == self.model.time + 1]

        current_locations = copy(self.model.movement.location.values[agents_leaving])
        self.model.step()
        new_locations = self.model.movement.location.values[agents_leaving]
        assert (new_locations == current_locations).sum() == 0, (
            f"{(new_locations == current_locations).sum()} of {len(current_locations)}"
            " expected agents did not relocate."
        )
