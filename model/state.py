from enum import unique, auto, IntEnum
from optabm.state import StateEnum


@unique
class LifeState(StateEnum):
    ALIVE = auto()
    DEAD = auto()


class AgeGroup(IntEnum):
    AGE0 = auto()
    AGE1 = auto()
    AGE2 = auto()


class LocationCategories(IntEnum):
    COMMUNITY = auto()
    HOSPITAL = auto()
    LT = auto()
    NH = auto()


LifeState.id = "LIFE"
AgeGroup.id = "AGE"
LocationCategories.id = "LocationCategories"


class Empty:
    """An empty state to house extra arrays or dictionaries"""

    def __init__(self):
        pass
