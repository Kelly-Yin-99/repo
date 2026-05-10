import unittest
from bayes_factor import BayesFactor


A = 0.4999
B = 0.5001


def bf(n=10, k=5, a=A, b=B):
    return BayesFactor(n, k, a, b)


class TestBayesFactor(unittest.TestCase):

    ## shared setup 
    def setUp(self):
        self.model = bf()

    ## Input and state check
    
    # check constructor stores values correctly
    def test_keeps_the_values_given_to_constructor(self):
        self.assertEqual(self.model.n, 10)
        self.assertEqual(self.model.k, 5)
        self.assertAlmostEqual(self.model.a, A)
        self.assertAlmostEqual(self.model.b, B)

    # invalid inputs for n and k
    def test_constructor_rejects_bad_counts(self):
        with self.assertRaisesRegex(ValueError, "n must be an integer"):
            BayesFactor(10.2, 5, A, B)

        with self.assertRaisesRegex(ValueError, "k must be an integer"):
            BayesFactor(10, 5.2, A, B)

        with self.assertRaisesRegex(ValueError, "n must be non-negative"):
            BayesFactor(-1, 0, A, B)

        with self.assertRaisesRegex(ValueError, "k must be non-negative"):
            BayesFactor(10, -1, A, B)

        # impossible binomial case
        with self.assertRaisesRegex(ValueError, "k cannot be greater than n"):
            BayesFactor(5, 6, A, B)

    # invalid spike interval
    def test_constructor_rejects_bad_interval(self):
        with self.assertRaisesRegex(ValueError, "invalid interval"):
            BayesFactor(10, 5, 0.7, 0.2)

        with self.assertRaisesRegex(ValueError, "invalid interval"):
            BayesFactor(10, 5, -0.1, 0.5)

        with self.assertRaisesRegex(ValueError, "invalid interval"):
            BayesFactor(10, 5, 0.4, 1.2)
            
            

    ## API check
    
    # API should exist
    def test_required_methods_are_available(self):
        self.assertTrue(callable(self.model.likelihood))
        self.assertTrue(callable(self.model.evidence_slab))
        self.assertTrue(callable(self.model.evidence_spike))
        self.assertTrue(callable(self.model.bayes_factor))

    # return types
    def test_main_methods_return_numbers(self):
        self.assertIsInstance(self.model.likelihood(0.5), float)
        self.assertIsInstance(self.model.evidence_slab(), float)
        self.assertIsInstance(self.model.evidence_spike(), float)
        self.assertIsInstance(self.model.bayes_factor(), float)
        

    ##  Math check
    
    # likelihood sanity check
    def test_likelihood_has_known_value_at_half(self):
        self.assertAlmostEqual(self.model.likelihood(0.5), 252 / 1024)

    # known closed-form result for slab
    def test_slab_has_simple_form(self):
        model = bf(n=10, k=3)
        self.assertAlmostEqual(model.evidence_slab(), 1 / 11, places=6)

    # slab should not depend on k
    def test_slab_does_not_depend_on_k(self):
        m1 = bf(n=10, k=3)
        m2 = bf(n=10, k=7)
        self.assertAlmostEqual(m1.evidence_slab(), m2.evidence_slab(), places=6)

    # spike behaves like point mass around 0.5
    def test_spike_behaves_like_point_mass(self):
        model = bf(n=10, k=5)
        self.assertAlmostEqual(
            model.evidence_spike(),
            model.likelihood(0.5),
            places=4
        )

    # if spike covers full range, equals slab
    def test_full_interval_makes_spike_and_slab_match(self):
        model = bf(n=10, k=5, a=0, b=1)
        self.assertAlmostEqual(
            model.evidence_spike(),
            model.evidence_slab(),
            places=6
        )

    # same prior , BF = 1
    def test_same_prior_gives_bayes_factor_one(self):
        model = bf(n=10, k=5, a=0, b=1)
        self.assertAlmostEqual(model.bayes_factor(), 1.0, places=6)
        

    ## Error behavior check
    
    def test_likelihood_rejects_bad_theta(self):
        with self.assertRaisesRegex(ValueError, "theta must be numeric"):
            self.model.likelihood("bad")

        with self.assertRaisesRegex(ValueError, "theta must be between 0 and 1"):
            self.model.likelihood(-0.1)

        with self.assertRaisesRegex(ValueError, "theta must be between 0 and 1"):
            self.model.likelihood(1.1)

    
    ## Robustness and edge cases check 
    # obvious edge cases for likelihood
    def test_likelihood_handles_obvious_edges(self):
        self.assertEqual(bf(10, 5).likelihood(0), 0)
        self.assertEqual(bf(10, 5).likelihood(1), 0)
        self.assertAlmostEqual(bf(0, 0).likelihood(0.5), 1.0)

        # evidence should not be negative
    def test_evidence_values_are_not_negative(self):
        self.assertGreaterEqual(self.model.evidence_slab(), 0)
        self.assertGreaterEqual(self.model.evidence_spike(), 0)

    # should not crash at edges
    def test_bayes_factor_runs_for_extreme_counts(self):
        low = bf(n=10, k=0)
        high = bf(n=10, k=10)

        self.assertIsInstance(low.bayes_factor(), float)
        self.assertIsInstance(high.bayes_factor(), float)
        

    ## Model behavior check

    # balanced data should favor spike
    def test_balanced_data_favors_the_spike(self):
        model = bf(n=100, k=50)
        self.assertGreater(model.bayes_factor(), 1)

    # extreme data should favor slab
    def test_far_from_half_favors_the_slab(self):
        model = bf(n=100, k=80)
        self.assertLess(model.bayes_factor(), 1)

    

    ## intentionally wrong TDD
    @unittest.expectedFailure
    def test_red_step_example(self):
        model = bf(n=10, k=5, a=0, b=1)
        self.assertAlmostEqual(model.bayes_factor(), 2.0)


if __name__ == "__main__":
    unittest.main()
