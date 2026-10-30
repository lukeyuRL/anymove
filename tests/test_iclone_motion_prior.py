"""Empirical motion prior training/denoising contract, pure NumPy tests."""
import json
import pathlib
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from mocap.motion_prior import train_prior, denoise_euler, PROFILE
    from mocap.denoise_motion import quaternion_matrices


@unittest.skipUnless(np is not None, "NumPy optional in lightweight CI")
class MotionPriorTests(unittest.TestCase):
    def make_sequences(self, folder):
        files = []
        frames, bones = 140, ["pelvis", "knee_L", "knee_R"]
        t = np.arange(frames, dtype=float)
        for k in (0, 1, 2):
            e = np.zeros((frames, len(bones), 3))
            e[:, 1, 0] = .3*np.sin(.14*t+k*.2)
            e[:, 2, 1] = .16*np.sin(.21*t+k*.3)
            path = pathlib.Path(folder)/("clean_%d.json" % k)
            path.write_text(json.dumps({
                "bones": bones, "fps": 24.0,
                "euler": e.tolist()
            }), encoding="utf-8")
            files.append(path)
        manifest = pathlib.Path(folder)/"manifest.json"
        manifest.write_text(json.dumps({
            "version": 1,
            "sequences": [{"path": p.name,
                           "source_kind": "synthetic_ground_truth_animation",
                           "verified": True} for p in files]
        }), encoding="utf-8")
        return manifest, files

    def test_empirical_prior_reduces_synthetic_high_frequency_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, files = self.make_sequences(tmp)
            prior = train_prior(manifest)
            self.assertEqual(prior["profile"], PROFILE)
            self.assertEqual(len(prior["source_sequences"]), 3)
            self.assertEqual(prior["bones"], ["pelvis", "knee_L", "knee_R"])
            base = np.asarray(json.loads(files[0].read_text())["euler"])
            rng = np.random.default_rng(2)
            noisy = base.copy()
            noisy[:, 1:, :] += rng.normal(0., .055, noisy[:, 1:, :].shape)
            result = denoise_euler(noisy, prior)
            before = np.mean((noisy[:, 1:]-base[:, 1:])**2)
            after = np.mean((result[:, 1:]-base[:, 1:])**2)
            self.assertLess(after, before)
            np.testing.assert_array_equal(result[:, 0], noisy[:, 0])
            result_zero = denoise_euler(noisy, prior, strength=0)
            np.testing.assert_array_equal(result_zero, noisy)

    def test_limits_are_respected_and_order_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self.make_sequences(tmp)
            prior = train_prior(manifest)
            e = np.full((36, 3, 3), .4)
            limits = np.zeros((3, 3, 2))
            limits[..., 0], limits[..., 1] = -10., 10.
            got = denoise_euler(e, prior, limits)
            self.assertTrue((got[:, 1:] <= np.radians(10)+1e-9).all())
            np.testing.assert_array_equal(got[:, 0], e[:, 0])
            with self.assertRaisesRegex(ValueError, "Euler shape"):
                denoise_euler(np.zeros((36, 4, 3)), prior)

    def test_refuses_unverified_motion_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self.make_sequences(tmp)
            config = json.loads(manifest.read_text())
            config["sequences"][0]["verified"] = False
            manifest.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "verified"):
                train_prior(manifest)
            config["sequences"][0]["verified"] = True
            config["sequences"][0]["source_kind"] = "unreviewed_ai_solver_output"
            manifest.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "verified"):
                train_prior(manifest)

    def test_root_quaternion_rotation_matrix(self):
        q = np.array([[1., 0., 0., 0.],
                      [np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)]])
        matrices = quaternion_matrices(q)
        np.testing.assert_allclose(matrices[0], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(matrices[1] @ [1, 0, 0], [0, 1, 0],
                                   atol=1e-12)
        with self.assertRaisesRegex(ValueError, "zero"):
            quaternion_matrices(np.zeros((2, 4)))


if __name__ == "__main__":
    unittest.main()
