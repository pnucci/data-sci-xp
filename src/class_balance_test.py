import unittest
import numpy as np

from class_balance import balance_oversample


class TestClassBalance(unittest.TestCase):

    def test_raises_input_mismatch(self):
        self.assertRaises(ValueError, balance_oversample, [], [1])

    def test_balances_classes_correctly(self):
        x = [135, 31576, 3134, 26442, 486, 3587]
        y = np.array([10, 20, 20, 30, 30, 30])
        _, counts_before = np.unique(y, return_counts=True)

        x2, y2 = balance_oversample(x, y)
        _, counts_after = np.unique(y2, return_counts=True)

        def is_balanced(value):
            return value == max(counts_before)
        for count_for_class in counts_after:
            self.assertTrue(is_balanced(count_for_class))
