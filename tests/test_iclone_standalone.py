"""Standalone probe must produce a JSON report without iClone or our package."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


class StandaloneDiagnosticsTest(unittest.TestCase):
    def test_report_persists_before_rlpy_import(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        src = root / "diag_iclone.py"
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env["TEMP"] = directory
            env["TMP"] = directory
            env["TMPDIR"] = directory
            p = subprocess.run(
                [sys.executable, str(src)],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            report_file = pathlib.Path(directory) / "anymove_iclone_probe.json"
            self.assertTrue(report_file.exists(), p.stdout + p.stderr)
            with open(report_file, encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["build"], "P0-STANDALONE-20261028-01")
            self.assertEqual(report["status"], "completed_with_errors")
            self.assertEqual(report["errors"][0]["stage"], "import_rlpy")
            self.assertEqual(report["steps"][0]["status"], "failed")
            self.assertTrue(report["script_file"].endswith("diag_iclone.py"))


if __name__ == "__main__":
    unittest.main()
