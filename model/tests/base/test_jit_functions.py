from unittest.case import TestCase

import numpy as np
import pytest
from src.jit_functions import assign_conditions, first_true_value, sum_true, update_community_probability


@pytest.mark.usefixtures("model")
class TestJitFunctions(TestCase):
    @staticmethod
    def test_assign_conditions():
        """Concurrent conditions are assigned on initialization. Make sure the appropriate amount is assigned"""
        ages = np.array([0] * 5000 + [1] * 5000 + [2] * 5000)
        concurrent_conditions = assign_conditions(ages, np.random.rand(len(ages)))

        # Those ages 1 (50-64) should be assigned concurrent conditions 23.74% of the time
        age_1 = concurrent_conditions[ages == 1]
        assert np.isclose(sum(age_1) / len(age_1), 0.2374, atol=0.02)

        # Those ages 2 (65+) should be assigned concurrent conditions 54.97% of the time
        age_2 = concurrent_conditions[ages == 2]
        assert np.isclose(sum(age_2) / len(age_2), 0.5497, atol=0.02)

    def test_update_community_probability(self):
        """Community probabilities are updated and movement should be based on concurrent conditions"""
        concurrent_conditions = np.array([1] * self.model.population.shape[0])

        # Probability of movement before update:
        age_1 = self.model.movement.community_to_hospital_probabilities[self.model.age_group == 1]
        age_1_before = age_1.mean()
        age_2 = self.model.movement.community_to_hospital_probabilities[self.model.age_group == 2]
        age_2_before = age_2.mean()

        new_probabilities = update_community_probability(
            cp=self.model.movement.community_to_hospital_probabilities,
            age=self.model.age_group,
            cc=concurrent_conditions,
        )

        # After Updates: Probabilities should go up
        age_1 = new_probabilities[self.model.age_group == 1]
        assert np.isclose(age_1.mean() / age_1_before, 2.316, atol=0.01)
        age_2 = new_probabilities[self.model.age_group == 2]
        assert np.isclose(age_2.mean() / age_2_before, 1.437, atol=0.01)

    @staticmethod
    def test_first_true_value():
        a = np.array([False, False, False, True])
        assert first_true_value(a) == (True, 3)

        b = np.array([True, False, False, False])
        assert first_true_value(b) == (True, 0)

        c = np.array([False, False])
        assert first_true_value(c) == (False, -1)

    @staticmethod
    def test_first_true_value_nonbool():
        with pytest.raises(TypeError):
            d = np.array([1, 2, 3, np.nan])  # non-boolean array
            first_true_value(d)

    @staticmethod
    def test_sum_true():
        size = 1000
        a = np.full(size, True)
        assert sum_true(a) == size

        b = np.array([True, False, True, False])
        assert sum_true(b) == 2
