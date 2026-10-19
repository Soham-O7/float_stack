import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from datadiet.measure import Assumptions, account, train_and_measure
from measurement import measured_account, measure_callable


class MeasurementTests(unittest.TestCase):
    def test_fallback_matches_existing_account_contract(self):
        assumptions = Assumptions(power_w=100, grid_kg_per_kwh=0.5, rupees_per_hour=36)
        result = measured_account(60, assumptions)
        self.assertEqual(set(result), {"kwh", "gco2", "rupees"})
        self.assertEqual(result, account(60, assumptions))

    def test_measured_energy_and_emissions_keep_same_keys(self):
        assumptions = Assumptions(rupees_per_hour=36)
        result = measured_account(60, assumptions, energy_kwh=0.02, emissions_kg=0.01)
        self.assertEqual(set(result), {"kwh", "gco2", "rupees"})
        self.assertAlmostEqual(result["kwh"], 0.02)
        self.assertAlmostEqual(result["gco2"], 10.0)
        self.assertAlmostEqual(result["rupees"], 0.6)

    def test_measure_callable_marks_fallback_when_codecarbon_unavailable(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict(sys.modules, {"codecarbon": None}):
                result, measurement = measure_callable(
                    lambda: "ok", output_dir=output_dir, run_id="fallback-test"
                )
        self.assertEqual(result, "ok")
        self.assertEqual(measurement.source, "estimate_fallback")
        self.assertEqual(set(measurement.values), {"kwh", "gco2", "rupees"})

    def test_measure_callable_uses_codecarbon_energy_and_emissions(self):
        class FakeTracker:
            def __init__(self, **kwargs):
                self.final_emissions_data = SimpleNamespace(energy_consumed=0.025)

            def start(self):
                return None

            def stop(self):
                return 0.012

        fake_module = SimpleNamespace(OfflineEmissionsTracker=FakeTracker)
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict(sys.modules, {"codecarbon": fake_module}):
                result, measurement = measure_callable(
                    lambda: 42, output_dir=output_dir, run_id="measured-test"
                )
        self.assertEqual(result, 42)
        self.assertEqual(measurement.source, "codecarbon")
        self.assertAlmostEqual(measurement.values["kwh"], 0.025)
        self.assertAlmostEqual(measurement.values["gco2"], 12.0)


    def test_train_and_measure_records_codecarbon_source_for_each_strategy(self):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        class FakeTracker:
            def __init__(self, **kwargs):
                self.final_emissions_data = SimpleNamespace(energy_consumed=0.0001)

            def start(self):
                return None

            def stop(self):
                return 0.00002

        fake_module = SimpleNamespace(OfflineEmissionsTracker=FakeTracker)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] > 0).astype(int)

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict(sys.modules, {"codecarbon": fake_module}):
                output = train_and_measure(
                    X[:45], y[:45], X[45:], y[45:],
                    keep_frac=0.4,
                    make_model=lambda: RandomForestClassifier(
                        n_estimators=3, random_state=0, n_jobs=1
                    ),
                    measurement_output_dir=output_dir,
                    require_codecarbon=True,
                    run_id_prefix="unit-test",
                )

        self.assertEqual(len(output["table"]), 3)
        self.assertEqual(
            [row["measurement_source"] for row in output["table"]],
            ["codecarbon", "codecarbon", "codecarbon"],
        )
        for row in output["table"]:
            self.assertGreaterEqual(row["kwh"], 0.0)
            self.assertGreaterEqual(row["gco2"], 0.0)

    def test_require_codecarbon_raises_instead_of_falling_back(self):
        called = []
        with tempfile.TemporaryDirectory() as output_dir:
            with patch.dict(sys.modules, {"codecarbon": None}):
                with self.assertRaises(RuntimeError):
                    measure_callable(
                        lambda: called.append(True),
                        output_dir=output_dir,
                        allow_fallback=False,
                    )
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
