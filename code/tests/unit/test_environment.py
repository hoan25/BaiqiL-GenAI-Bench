"""Bootstrap tests runnable before pytest is installed."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.check_environment import check_module, collect_environment


class EnvironmentTests(unittest.TestCase):
    def test_import_and_required_symbol(self):
        with patch("scripts.check_environment.importlib.import_module", return_value=SimpleNamespace(__version__="1", CLIPModel=object())):
            self.assertEqual(check_module("fixture", ("CLIPModel",)), {"status": "PASS", "version": "1"})

    def test_missing_dependency(self):
        with patch("scripts.check_environment.importlib.import_module", side_effect=ModuleNotFoundError("missing")):
            self.assertEqual(check_module("fixture")["status"], "FAIL")

    def test_missing_required_symbol(self):
        with patch("scripts.check_environment.importlib.import_module", return_value=SimpleNamespace()):
            self.assertEqual(check_module("fixture", ("CLIPModel",))["status"], "FAIL")

    def test_pip_conflict_fails_overall_check(self):
        with patch("scripts.check_environment.check_module", return_value={"status": "PASS"}), patch("scripts.check_environment.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="conflict", stderr="")):
            result = collect_environment()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["checks"]["pip_check"]["output"], "conflict")

    def test_startup_error_cannot_pass_with_zero_exit_code(self):
        with patch("scripts.check_environment.check_module", return_value={"status": "PASS"}), patch("scripts.check_environment.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="No broken requirements found.", stderr="Error in sitecustomize: missing wrapt")):
            self.assertEqual(collect_environment()["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
