import unittest
from aox_g3.quality_display import healing_result


class QualityDisplayTests(unittest.TestCase):
    def test_execution_success_does_not_hide_quality_failure(self):
        result = healing_result({"stages": {"heal": {"status": "ok"}},
                                 "numbers": {"closed": False, "valid": False,
                                             "floating_caps": 11, "free_boundaries_measured": 14}})
        self.assertEqual(result["execution_status"], "ok")
        self.assertIs(result["closed"], False)
        self.assertIs(result["valid"], False)
        self.assertEqual(result["floating_caps"], 11)
        self.assertEqual(result["free_boundaries_measured"], 14)

    def test_missing_values_are_unknown_not_success_or_failure(self):
        result = healing_result({})
        self.assertEqual(result["execution_status"], "unknown")
        self.assertIsNone(result["closed"])
        self.assertIsNone(result["valid"])
        self.assertIsNone(result["floating_caps"])

    def test_true_and_zero_are_preserved(self):
        result = healing_result({"stages": {"heal": {"status": "failed"}},
                                 "numbers": {"closed": True, "valid": True, "floating_caps": 0}})
        self.assertEqual(result["execution_status"], "failed")
        self.assertIs(result["closed"], True)
        self.assertIs(result["valid"], True)
        self.assertEqual(result["floating_caps"], 0)


if __name__ == "__main__":
    unittest.main()
