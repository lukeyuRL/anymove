"""Offline tests for upstream Dragon Solver vs Blender and detections."""
import copy
import math
import unittest
from iclone_adapter.source_pose_diagnose import diagnose


class SourcePoseTests(unittest.TestCase):
    def setUp(self):
        self.names=["hips","head","fpaw_L","fpaw_R"]
        self.rows=[[[0.,0.,.9],[0.,-.8,1.7+.01*f],
                    [.3,-1.,.1],[-.3,-1.,.1]] for f in range(4)]
        self.motion={"fps":24.0,"joint_names":self.names,"joints":self.rows,
                     "euler":[[[0.,0.,0.]]*2 for _ in range(4)],
                     "root_quat":[[1.,0.,0.,0.] for _ in range(4)]}
        self.rig={"bones":[{"name":"pelvis","matrix_local":[
            [1.,0.,0.,0.],[0.,-.948683,-.316228,.32],
            [0.,.316228,-.948683,.86],[0.,0.,0.,1.]]}]}
        self.blender={"fps":24.,"names":["head","hips","fpaw_L","fpaw_R"],
                      "joints":[[row[1],row[0],row[2],row[3]]
                                for row in self.rows]}
        self.layout={"W_FRONT":640,"W_SIDE":640,
                     "M_PER_PX":.01,"GROUND_V":630}
        self.det={"fps":24.,"names":self.names,"front_uv":[],
                  "side_uv":[],"front_conf":[],"side_conf":[]}
        for row in self.rows:
            a=[];b=[]
            for x,y,z in row:
                v=630-100*z
                a.append([320+100*x,v])
                b.append([320-100*y,v])
            self.det["front_uv"].append(a)
            self.det["side_uv"].append(b)
            self.det["front_conf"].append([1.]*len(row))
            self.det["side_conf"].append([1.]*len(row))

    def compare(self,motion=None,blender=None,det=None):
        return diagnose(
            self.motion if motion is None else motion,self.rig,
            blender=self.blender if blender is None else blender,
            detections=self.det if det is None else det,
            layout=self.layout)

    def test_same_solver_and_blender_pose_pass(self):
        report=self.compare()
        self.assertEqual(report["result"],
            "solver_blender_match_detected_2D_check_3D_visual_and_skin")
        self.assertEqual(report["solver_to_blender_cm"]["error_cm"]["max"],0.)
        self.assertLess(report["pelvis_root_world_orientation"][
            "world_up_tilt_deg"]["max"],1e-4)

    def test_blender_world_pose_mismatch_is_localized(self):
        wrong=copy.deepcopy(self.blender)
        wrong["joints"][2][0][0]+=.20
        report=self.compare(blender=wrong)
        self.assertEqual(report["result"],
                         "blender_pose_or_rig_differs_from_solver")
        self.assertAlmostEqual(report["solver_to_blender_cm"][
            "error_cm"]["max"],20.)

    def test_solver_leans_unlike_source_detection(self):
        wrong=copy.deepcopy(self.motion)
        for row in wrong["joints"]:
            row[1][0]+=.7
        # The Blender reference agrees with the modified Solver here,
        # isolating the source-DINO joint mismatch rather than Blender.
        blender=copy.deepcopy(self.blender)
        for frame in blender["joints"]:
            frame[0][0]+=.7
        report=self.compare(motion=wrong,blender=blender)
        self.assertEqual(report["result"],
                         "solver_deviates_from_detected_video_joints")
        self.assertGreater(report["source_detection_fit_px"][
            "front_lean_disagreement_deg"]["p95"],10.)

    def test_world_root_basis_conjugation(self):
        wrong=copy.deepcopy(self.motion)
        angle=math.radians(45.)
        # Pelvis local X points along world X; local Y is tilted.
        wrong["root_quat"][1]=[
            math.cos(angle*.5),math.sin(angle*.5),0.,0.]
        report=self.compare(motion=wrong)
        self.assertAlmostEqual(report["pelvis_root_world_orientation"][
            "world_up_tilt_deg"]["max"],45.,places=3)

    def test_refuse_unrelated_motion_blender_or_detection(self):
        bad=copy.deepcopy(self.blender)
        bad["joints"].pop()
        with self.assertRaisesRegex(ValueError,"NOT the same motion"):
            self.compare(blender=bad)
        det=copy.deepcopy(self.det)
        det["names"][0]="wrong"
        with self.assertRaisesRegex(ValueError,"EXACT"):
            self.compare(det=det)
        det=copy.deepcopy(self.det)
        det["front_uv"].pop()
        with self.assertRaisesRegex(ValueError,"EXACT"):
            self.compare(det=det)

    def test_no_blender_or_source_is_not_a_false_success(self):
        report=diagnose(self.motion,self.rig,layout=self.layout)
        self.assertEqual(report["result"],"needs_matching_blender_joints_json")
        self.assertEqual(report["source_detection_fit_px"]["status"],
                         "not_available")


if __name__=="__main__":
    unittest.main()
