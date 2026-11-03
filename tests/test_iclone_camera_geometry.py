"""Arbitrary horizontal camera-angle contract, solver and Blender geometry QA.

Run: python -m unittest tests.test_iclone_camera_geometry -v
No GPU or Blender is required for these tests.
"""
import hashlib
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from mocap import layout, solve
from mocap.camera_geometry import (
    pair_condition, project_orthographic, triangulate_orthographic,
    validate_pair, verify_record_angles, verify_reference_file, resolve_angles,
)
from mocap.layout_guard import verify_layout_for_rig


class CameraGeometryTests(unittest.TestCase):
    @staticmethod
    def samples():
        rng = np.random.default_rng(20261103)
        out = rng.normal(size=(7, 22, 3)) * [0.5, 1.1, 0.35]
        out[..., 2] += 1.2
        return out

    def project_pair(self, points, angles):
        f = project_orthographic(points, angles[0], 640, 720, .006875, .75)
        s = project_orthographic(points, angles[1], 640, 720, .006875, .75)
        return f, s

    def reconstruct(self, angles, f=None, s=None):
        points = self.samples()
        if f is None:
            f, s = self.project_pair(points, angles)
        t, j, _ = points.shape
        recovered = triangulate_orthographic(
            f, s, np.ones((t, j)), np.ones((t, j)),
            front_width=640, side_width=640, height=720,
            metres_per_pixel=.006875, z_center=.75,
            front_angle_deg=angles[0], side_angle_deg=angles[1],
        )
        return points, recovered

    def test_original_front_side_exact_compatibility(self):
        points = self.samples()
        front, side = self.project_pair(points, (0, 90))
        expected_front_u = 320 + points[..., 0] / .006875
        expected_side_u = 320 - points[..., 1] / .006875
        np.testing.assert_allclose(front[..., 0], expected_front_u, atol=1e-10)
        np.testing.assert_allclose(side[..., 0], expected_side_u, atol=1e-10)
        original, recovered = self.reconstruct((0, 90), front, side)
        np.testing.assert_allclose(recovered, original, atol=1e-12)

    def test_arbitrary_right_and_left_front_cameras(self):
        for angles in ((315, 45), (-45, 45), (45, 135), (20, 100),
                       (5, 65), (330, 90), (0, 120)):
            with self.subTest(angles=angles):
                original, recovered = self.reconstruct(angles)
                np.testing.assert_allclose(recovered, original, atol=1e-12)

    def test_noisy_joint_reconstruction_is_finite(self):
        points = self.samples()
        a = (315, 45)
        front, side = self.project_pair(points, a)
        rng = np.random.default_rng(3)
        front += rng.normal(0, .5, front.shape)
        side += rng.normal(0, .5, side.shape)
        _, recovered = self.reconstruct(a, front, side)
        self.assertTrue(np.isfinite(recovered).all())
        self.assertLess(np.linalg.norm(points - recovered, axis=-1).mean(), .02)

    def test_orthogonal_geometry_and_degenerate_rejection(self):
        self.assertAlmostEqual(pair_condition(315, 45), 1.0, places=12)
        for bad in ((45, 225), (0, 180), (20, 20), (0, 1), (0, 179)):
            with self.subTest(angles=bad):
                with self.assertRaisesRegex(ValueError, "DEGENERATE_CAMERA_PAIR"):
                    validate_pair(*bad)
        for invalid in ((float("nan"), 90), (0, float("inf")), (True, 90)):
            with self.assertRaises(ValueError):
                validate_pair(*invalid)

    def test_torch_projection_matches_numpy_and_backpropagates(self):
        points = self.samples()
        for angles in ((0, 90), (315, 45), (45, 135)):
            with self.subTest(angles=angles):
                with mock.patch.object(layout, "FRONT_AZIMUTH_DEG", angles[0]), mock.patch.object(
                    layout, "SIDE_AZIMUTH_DEG", angles[1]
                ):
                    t = torch.tensor(points, dtype=torch.float64, requires_grad=True)
                    f, s = solve.project(t)
                    ef = project_orthographic(points, angles[0], layout.W_FRONT, layout.H,
                                               layout.M_PER_PX, layout.Z_CENTER)
                    es = project_orthographic(points, angles[1], layout.W_SIDE, layout.H,
                                               layout.M_PER_PX, layout.Z_CENTER)
                    np.testing.assert_allclose(f.detach(), ef, atol=1e-10)
                    np.testing.assert_allclose(s.detach(), es, atol=1e-10)
                    (f.square().mean() + s.square().mean()).backward()
                    self.assertTrue(torch.isfinite(t.grad).all())

    def test_height_weight_default_preserves_legacy_and_is_symmetric_for_oblique(self):
        with mock.patch.object(layout, 'FRONT_AZIMUTH_DEG', 0.0), mock.patch.object(
            layout, 'SIDE_AZIMUTH_DEG', 90.0
        ):
            self.assertEqual(solve.effective_front_height_weight(), 0.25)
        with mock.patch.object(layout, 'FRONT_AZIMUTH_DEG', 315.0), mock.patch.object(
            layout, 'SIDE_AZIMUTH_DEG', 45.0
        ):
            self.assertEqual(solve.effective_front_height_weight(), 1.0)
            self.assertEqual(solve.effective_front_height_weight(0.6), 0.6)

    def test_solver_triangulation_matches_shared_geometry(self):
        points = self.samples()
        for angles in ((0, 90), (315, 45), (45, 135)):
            with self.subTest(angles=angles):
                f = project_orthographic(points, angles[0], layout.W_FRONT, layout.H,
                                         layout.M_PER_PX, layout.Z_CENTER)
                s = project_orthographic(points, angles[1], layout.W_SIDE, layout.H,
                                         layout.M_PER_PX, layout.Z_CENTER)
                d = {"front_uv": f, "side_uv": s,
                     "front_conf": np.ones(points.shape[:2]),
                     "side_conf": np.ones(points.shape[:2])}
                with mock.patch.object(layout, "FRONT_AZIMUTH_DEG", angles[0]), mock.patch.object(
                    layout, "SIDE_AZIMUTH_DEG", angles[1]
                ):
                    np.testing.assert_allclose(solve.triangulate(d), points, atol=1e-12)

    def test_pixel_height_is_shared_for_all_level_angles(self):
        points = self.samples()
        f, s = self.project_pair(points, (315, 45))
        np.testing.assert_allclose(f[..., 1], s[..., 1], atol=1e-12)

    def test_detections_and_reader_provenance_require_matching_angles(self):
        self.assertEqual(verify_record_angles(None, (0, 90), "legacy"), "legacy_default_0_90")
        self.assertEqual(verify_record_angles([-45, 405], (315, 45), "dataset"), "camera_angles_verified")
        for record, expected, marker in (
            (None, (315, 45), "CAMERA_ANGLE_METADATA_MISSING"),
            ([0, 90], (315, 45), "CAMERA_ANGLE_MISMATCH"),
            ([315], (315, 45), "CAMERA_ANGLE_METADATA_INVALID"),
        ):
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, marker):
                    verify_record_angles(record, expected, "detections")

    def test_angle_environment_override(self):
        self.assertEqual(resolve_angles(0, 90, '315,45'), (315.0, 45.0))
        self.assertEqual(resolve_angles(45, 135, '-45,45'), (315.0, 45.0))
        with self.assertRaisesRegex(ValueError, 'MOCAP_CAMERA_AZIMUTHS'):
            resolve_angles(0, 90, '315')
        with self.assertRaisesRegex(ValueError, 'DEGENERATE_CAMERA_PAIR'):
            resolve_angles(0, 90, '45,225')

    def test_reference_image_sha_and_angle_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = pathlib.Path(tmp) / 'ref.png'
            ref.write_bytes(b'controlled-ref-image')
            self.assertEqual(verify_reference_file(ref, (0, 90)), 'legacy_default_0_90')
            with self.assertRaisesRegex(ValueError, 'CAMERA_ANGLE_METADATA_MISSING'):
                verify_reference_file(ref, (315, 45))
            manifest = ref.with_suffix('.camera.json')
            manifest.write_text(json.dumps({'camera_azimuth_deg': [315, 45],
                                            'reference_sha256': hashlib.sha256(ref.read_bytes()).hexdigest()}))
            self.assertEqual(verify_reference_file(ref, (315, 45)), 'camera_angles_verified')
            ref.write_bytes(b'modified-ref')
            with self.assertRaisesRegex(ValueError, 'CAMERA_REFERENCE_SHA_MISMATCH'):
                verify_reference_file(ref, (315, 45))

    def test_layout_guard_checks_new_angle_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig = pathlib.Path(tmp) / "assets" / "dragon" / "rigged" / "rig_info.json"
            rig.parent.mkdir(parents=True)
            rig.write_text("{}", encoding="utf-8")
            cfg = rig.parent.parent / "creature.json"
            cfg.write_text(json.dumps({"name": "dragon", "layout": {
                "W_FRONT": 640, "W_SIDE": 640, "M_PER_PX": .006875, "GROUND_V": 630,
                "FRONT_AZIMUTH_DEG": 315, "SIDE_AZIMUTH_DEG": 45
            }}), encoding="utf-8")
            active = SimpleNamespace(W_FRONT=640, W_SIDE=640, M_PER_PX=.006875,
                                     GROUND_V=630, FRONT_AZIMUTH_DEG=-45,
                                     SIDE_AZIMUTH_DEG=45)
            self.assertEqual(verify_layout_for_rig(rig, active)["status"],
                             "camera_layout_verified_for_rig")
            active.FRONT_AZIMUTH_DEG = 0
            with self.assertRaisesRegex(RuntimeError, "FRONT_AZIMUTH_DEG"):
                verify_layout_for_rig(rig, active)
            active.SIDE_AZIMUTH_DEG = 90
            with mock.patch.dict('os.environ', {'MOCAP_CAMERA_AZIMUTHS': '0,90'}):
                self.assertEqual(verify_layout_for_rig(rig, active)['status'],
                                 'camera_layout_verified_for_rig')


if __name__ == "__main__":
    unittest.main()
