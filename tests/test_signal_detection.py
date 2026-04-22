import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from signal_detection import SignalDetection


class TestSignalDetection(unittest.TestCase):

    def test_hit_rate_basic(self):
        sdt = SignalDetection(hit=40, misses=10, false_alarms=5, correct_rejections=45)
        self.assertAlmostEqual(sdt.hit_rate(), 0.8)

    def test_false_alarm_rate_basic(self):
        sdt = SignalDetection(hit=40, misses=10, false_alarms=5, correct_rejections=45)
        self.assertAlmostEqual(sdt.false_alarm_rate(), 0.1)

    def test_d_prime_known_value(self):
        sdt = SignalDetection(hit=40, misses=10, false_alarms=5, correct_rejections=45)
        expected = 1.75
        self.assertAlmostEqual(sdt.d_prime(), expected, places=1)

    def test_criterion_known_value(self):
        sdt = SignalDetection(hit=40, misses=10, false_alarms=5, correct_rejections=45)
        expected = 0.3
        self.assertAlmostEqual(sdt.criterion(), expected, places=1)


    def test_negative_counts_raise(self):
        with self.assertRaises(ValueError):
            SignalDetection(hit=-1, misses=10, false_alarms=5, correct_rejections=45)

    def test_wrong_type_raise(self):
        with self.assertRaises(TypeError):
            SignalDetection(hit="a", misses=10, false_alarms=5, correct_rejections=45)

    def test_invalid_operator_argument(self):
        sdt = SignalDetection(10, 10, 10, 10)
        with self.assertRaises(TypeError):
            _ = sdt + 5

    def test_internal_consistency_after_add(self):
        s1 = SignalDetection(5, 10, 5, 10)
        s2 = SignalDetection(10, 10, 5, 15)

        s3 = s1 + s2
        self.assertEqual(s3.hits, 15)
        self.assertEqual(s3.misses, 20)
        self.assertEqual(s3.false_alarms, 10)
        self.assertEqual(s3.correct_rejections, 25)


    def test_add_operator(self):
        s1 = SignalDetection(5, 2, 5, 10)
        s2 = SignalDetection(10, 8, 5, 15)

        s3 = s1 + s2
        self.assertEqual(s3.hits, 15)
        self.assertEqual(s3.misses, 10)

    def test_sub_operator(self):
        s1 = SignalDetection(10, 10, 10, 10)
        s2 = SignalDetection(5, 5, 5, 5)

        s3 = s1 - s2
        self.assertEqual(s3.hits, 5)
        self.assertEqual(s3.misses, 5)

    def test_mul_operator(self):
        s = SignalDetection(10, 10, 10, 10)
        s2 = s * 2

        self.assertEqual(s2.hits, 20)
        self.assertEqual(s2.false_alarms, 20)

    def test_non_mutation(self):
        s1 = SignalDetection(10, 10, 10, 10)
        s2 = s1 * 2

        self.assertEqual(s1.hits, 10)
        self.assertNotEqual(s1.hits, s2.hits)

    def test_plot_sdt_returns_figure(self):
        sdt = SignalDetection(10, 10, 5, 15)
        fig = sdt.plot_sdt()

        self.assertIsInstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_roc_multiple_objects(self):
        s1 = SignalDetection(10, 10, 5, 15)
        s2 = SignalDetection(20, 5, 3, 17)

        fig = SignalDetection.plot_roc([s1, s2])
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
    def test_plot_roc_bounds(self):
        s1 = SignalDetection(10, 10, 5, 15)
        fig = SignalDetection.plot_roc([s1])

        ax = fig.axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        self.assertTrue(xlim[0] <= 0)
        self.assertTrue(xlim[1] >= 1)
        self.assertTrue(ylim[0] <= 0)
        self.assertTrue(ylim[1] >= 1)

        plt.close(fig)

    def test_edge_case_zero_counts(self):
        sdt = SignalDetection(0, 0, 0, 0)
        rate = sdt.hit_rate()

        self.assertTrue(np.isnan(rate) or rate == 0)


if __name__ == '__main__':
    unittest.main()