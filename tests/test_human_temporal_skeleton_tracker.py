"""P1 temporal/skeleton association contracts with synthetic 67-point rig."""
import copy, unittest
import numpy as np

from mocap.camera_geometry import project_orthographic
from mocap.temporal_skeleton_tracker import (
    SOURCE_SHA, SCHEMA, track_two_view, _validated_calibration,
)
from mocap.solve import apply_temporal_joint_guard
from iclone_adapter.motion_contract import validate_motion
from iclone_adapter.schema import ContractError, VERSION, IDENTITY

class TemporalSkeletonTrackerTests(unittest.TestCase):
    def setUp(self):
        self.names=["J%02d"%i for i in range(67)]
        self.parent_names={n:(self.names[i-1] if i>0 else None)
                           for i,n in enumerate(self.names)}
        self.parents=[-1]+list(range(66))
        self.frames=np.array([8,15,22,29,36],dtype=int)
        xyz=[]
        for t in range(len(self.frames)):
            a=np.zeros((67,3),float)
            a[0]=(0.03*t,0.02,1.05)
            for j in range(1,67):
                a[j]=a[j-1]+np.array([0.008,0.001,0.004])
            xyz.append(a)
        self.xyz=np.stack(xyz)
        self.front=project_orthographic(self.xyz,315,672,672,2.432802200317383/672,.8688581585884094)
        self.side=project_orthographic(self.xyz,45,672,672,2.432802200317383/672,.8688581585884094)
        geo=np.array([.008,.001,.004])
        rest=float(np.linalg.norm(geo))
        cam={}
        for slot,uv in (("front",self.front),("side",self.side)):
            gates=[]
            for j in range(67):
                delta=uv[:,j] if j==0 else uv[:,j]-uv[:,j-1]
                gates.append({"mean_delta_xy":np.mean(delta,axis=0).tolist(),
                              "length_gate_px":1000. if j==0 else 15.,
                              "velocity_gate_px_per_frame":12. if j==0 else 2.0})
            cam[slot]=gates
        self.cal={
          "schema":SCHEMA,"source_blend_sha256":SOURCE_SHA,
          "camera_azimuth_deg":[315,45],"image_width":672,"image_height":672,
          "metres_per_pixel":2.432802200317383/672,"z_center":.8688581585884094,
          "names":self.names,"parent_names":self.parent_names,
          "parent_indices":self.parents,"rest_lengths_m":[0]+[rest]*66,
          "mean_relative_xyz_m":[[0.,0.,0.]]+[geo.tolist()]*66,
          "calibration_clip_count":144,
          "cameras":cam,
        }

    def test_clean_sequence_retained(self):
        r=track_two_view(self.front,self.side,self.frames,self.cal,self.names)
        np.testing.assert_allclose(r["front_uv"],self.front,atol=1e-10)
        np.testing.assert_allclose(r["side_uv"],self.side,atol=1e-10)
        np.testing.assert_allclose(r["joints_world_m"],self.xyz,atol=1e-10)
        self.assertEqual(r["diagnostics"]["rejected_front"],0)
        self.assertEqual(r["diagnostics"]["rejected_side"],0)
        self.assertEqual(r["diagnostics"]["skeleton_repaired"],0)

    def test_single_view_extreme_finger_confusion_rejected(self):
        corrupt=self.front.copy()
        corrupt[2,30]+=[-210.,160.]
        r=track_two_view(corrupt,self.side,self.frames,self.cal,self.names)
        self.assertTrue(r["front_inferred"][2,30])
        self.assertEqual(r["front_reliability"][2,30],.2)
        self.assertFalse(r["side_inferred"][2,30])
        self.assertLess(np.linalg.norm(r["joints_world_m"][2,30]-self.xyz[2,30]),.04)

    def test_good_45_degree_view_not_rewritten_for_localized_error(self):
        corrupt=self.front.copy()
        corrupt[3,56]+=[190.,-110.]
        r=track_two_view(corrupt,self.side,self.frames,self.cal,self.names)
        self.assertFalse(r["side_inferred"][3,56])
        self.assertLess(np.linalg.norm(r["side_uv"][3,56]-self.side[3,56]),1e-8)

    def test_immutable_input_and_calibration(self):
        raw=self.front.copy()
        cal_before=copy.deepcopy(self.cal)
        _=track_two_view(raw,self.side,self.frames,self.cal,self.names)
        np.testing.assert_array_equal(self.front,raw)
        self.assertEqual(self.cal,cal_before)

    def test_solver_opt_in_confidence_and_source_preservation(self):
        src={
           'front_uv':self.front.copy(),
           'side_uv':self.side.copy(),
           'front_conf':np.ones((len(self.frames),67)),
           'side_conf':np.ones((len(self.frames),67))
        }
        src['front_uv'][2,30]+=[210,-120]
        fixed,report=apply_temporal_joint_guard(src,self.names,self.frames,self.cal)
        self.assertEqual(report['source_detections_preserved'],True)
        self.assertEqual(fixed['front_conf'][2,30],.2)
        self.assertEqual(src['front_conf'][2,30],1.)
        self.assertEqual(report['camera_azimuth_deg'],[315,45])
        self.assertGreater(report['tracker_diagnostics']['rejected_front'],0)
        self.assertLess(np.linalg.norm(fixed['front_uv'][2,30]-self.front[2,30]),30)

    def test_blender_solver_motion_never_passes_iclone_fk_writer_contract(self):
        fake_skeleton={
           'schema_version':VERSION,
           'reference':{'kind':'current_snapshot'},
           'bones':[{'id':'bone0','path':'/bone0','parent_id':None,
                     'snapshot_local_matrix':IDENTITY,
                     'snapshot_world_matrix':IDENTITY}]
        }
        blender_motion={
            'fps':24,'bones':['CC_Base_L_Hand'],
            'euler':[[[0.,0.,0.]]],
            'motion_space_warning':'BLENDER_LOCAL_EULER_NOT_ICLONE_FK_CHANNELS'
        }
        with self.assertRaisesRegex(ContractError,'expected explicit iClone FK'):
            validate_motion(blender_motion,fake_skeleton)

    def test_rejects_bad_frames_and_rig_provenance(self):
        with self.assertRaisesRegex(ValueError,"increasing"):
            track_two_view(self.front,self.side,[8,15,15,29,36],self.cal,self.names)
        bad=copy.deepcopy(self.cal)
        bad["camera_azimuth_deg"]=[0,90]
        with self.assertRaisesRegex(ValueError,"camera"):
            track_two_view(self.front,self.side,self.frames,bad,self.names)
        with self.assertRaisesRegex(ValueError,"names"):
            track_two_view(self.front,self.side,self.frames,self.cal,list(reversed(self.names)))
        with self.assertRaisesRegex(ValueError,"Reader UV"):
            track_two_view(self.front[:3],self.side,self.frames,self.cal,self.names)

if __name__=="__main__":
    unittest.main()
