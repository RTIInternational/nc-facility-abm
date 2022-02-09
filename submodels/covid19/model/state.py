from enum import auto

from optabm.state import StateEnum


class COVIDState(StateEnum):
    SUSCEPTIBLE = auto()
    ASYMPTOMATIC = auto()
    MILD = auto()
    SEVERE = auto()
    CRITICAL = auto()
    RECOVERED = auto()


class COVIDTest(StateEnum):
    NA = auto()
    TESTED = auto()
    NOTTESTED = auto()


class VaccinationStatus(StateEnum):
    NOTVACCINATED = auto()
    VACCINATED = auto()


class WorkerType(StateEnum):
    SINGLE_SITE_FULL_TIME = auto()
    SINGLE_SITE_PART_TIME = auto()
    MULTI_SITE = auto()
    CONTRACT = auto()
