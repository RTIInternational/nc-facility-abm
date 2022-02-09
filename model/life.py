from optabm.state import State
from optabm.state import LifeState
from optabm.storage import EventStorage


class Life(State):
    """This Life state is a simple value for Living or Dead. Agents in the HospitalABM can only die when leaving
    a facility.
    """

    def __init__(self, model):
        super().__init__(model=model, state_enum=LifeState)
        self.values.fill(LifeState.ALIVE.value)
        self.state_changes = EventStorage(column_names=["Time", "Unique_ID", "Location"])

    def death(self, unique_id: int):
        """The default death method for an agent is to move from alive to dead."""

        # Record the state change
        current_location = self.model.movement.location[unique_id]
        self.state_changes.record_event((self.model.time, unique_id, current_location))

        # The agent is now dead.
        self[unique_id] = LifeState.DEAD.value

        # If agent is not at home, send them home
        if current_location != self.model.nodes.community:
            self.model.movement.go_home(unique_id=unique_id, current_location=current_location)

    def step(self):
        pass

    @property
    def is_living(self):
        return self.values == LifeState.ALIVE
