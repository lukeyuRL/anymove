# AnyMove for iClone — P0/P1 experimental slice

Target branch: `iClone`. The original Blender/AnyMove pipeline stays untouched.

## Status (do not treat this as end-to-end capture)

- **P0 built:** read one selected rigged Avatar/Prop's Skin Bone hierarchy, native current-pose local/world transforms, supported SDK flags and export JSON.
- **P0 NOT verified:** true Skin Bind/Rest Pose, native-to-metre scale, per-asset animation control semantics and iClone 8/9 real runtime.
- **P1 built:** strictly typed in-scene FK **channel** motion, optional synthetic 48-frame sine test, safety preflight, new motion clip and attempted rollback.
- **P1 NOT verified:** end-to-end real SDK write/readback, clip range, FK/IK interaction, arbitrary Skeleton retargeting, AnyMove Blender Euler conversion, root motion and Generic Solver.
- DINOv2 and torch **must run in a separate Python 3.11 process**, not the iClone Python 3.8 host.

## Run in iClone 8 Python console

Clone the repository locally and select **exactly one** Avatar or rigged Prop in iClone.
In the iClone Python Console (use your actual clone path):

```python
import sys
sys.path.insert(0, r"F:\anymove")
from iclone_adapter.run_in_iclone import show
show()
```

1. **Inspect selected skeleton**: check `GetSkinBones`, `GetClipCount`, `AddClip`, `DeleteClip`, and other API capabilities, without modifying the scene.
2. **Export current-pose skeleton snapshot**: write `skeleton.json`. The reference is **current_snapshot**, not verified bind pose. Bones' local/world matrices are transposed from RLPy's row-vector convention and retain **unverified native units**. No scene changes.
3. **Create 48-frame diagnostic FK motion**: choose a skin bone; save `test_fk.json` with an X-channel sine curve, 24 fps. This is an *iClone channel-space* diagnostic, NOT AnyMove capture.
4. Open an expendable iClone scene with the **same selected avatar**, **zero existing motion clips**, **project 24 FPS**, **timeline at frame 0**. Use **Write FK motion**. The dialog explicitly requires acknowledgment of the scene mutation.
5. Inspect the result manually. A successful tool return is marked `keys_written_unverified`, not proven correct. Verify that the bone rotates around the intended X channel, moves over 48 frames, and the last frame returns to zero.

**Never** pass the upstream `mocap.solve` `motion.json` directly into the P1 writer. It contains Blender-specific local Euler XYZ + root data. FK Bone Rest-axis mapping and retargeting must be built and calibrated first. The included `examples/dragon_roar/det.npz` is useful for subsequent P2 testing, but is not a stand-in for the bone-channel conversion.

## Invariants

- P0 never modifies the scene. No claim that `GetBoneTPosePosition` is a complete bind matrix: the API documents a position only.
- The P1 writer requires a hierarchy signature from the **current** live scene and rejects any foreign/synthetic skeleton ID.
- Only `anymove.iclone.fk_channels/1` files can be written; no assumptions about arbitrary quaternion axes.
- P1 refuses any avatar that already has one or more Motion Clips. This is an intentional MVP safety constraint, not a product feature.
- A failed write tries to delete the newly created clip. If rollback fails the error must be treated as a potentially modified scene.
- Existing original AnyMove files, assets and its standard Python environment remain unchanged.

## Offline unit tests

```bash
python -m unittest discover -s tests -p "test_iclone_*.py" -v
```

Python 3.8 and 3.11 are targeted. Tests use mock RLPy to cover hierarchy, row-vector matrix conversion, FK channel writes and contract refusal; they **do not substitute for iClone runtime tests**.

## SDK references

- https://wiki.reallusion.com/IC8_Python_API:RLPy_RISkeletonComponent
- https://wiki.reallusion.com/IC_8_Python_API:End_Effector_Animation
- https://wiki.reallusion.com/IC_8_Python_API:Dealing_With_Custom_FPS
- https://wiki.reallusion.com/IC_Python_API:RLPy_RTransform

## Next engineering blockers

1. Verify true bind/rest matrices and per-bone axis conversion with real iClone avatars (P0 gate).
2. Verify FK channels and clip duration, readback and FK/IK override with real iClone runtime (P1 gate).
3. Implement a Generic Pose quaternion contract and calibrated Blender/AnyMove-to-iClone mapping, before consuming Dragon det.npz/solver motions.
4. Implement direct viewport dual-camera reference/dataset rendering, DINOv2 training/inference service and retargeting.

## Convert the supplied Dragon detection through AnyMove (external Python 3.11)

The original repo includes `examples/dragon_roar/det.npz`. Without DINOv2
training, run in the **external** AnyMove environment:

```bash
MOCAP_CREATURE=assets/dragon python -m mocap.solve \
  examples/dragon_roar/det.npz assets/dragon/rigged/rig_info.json \
  runs/dragon/clips/roar/motion.json
```

Do not load this Blender motion into the P1 writer directly. Instead, export
a snapshot from the SAME iClone scene and generate a mapping template:

```bash
python -m iclone_adapter.anymove_bridge map \
  runs/dragon/clips/roar/motion.json skeleton.json map.json
```

Inspect `map.json`. For each desired source Bone, set `target_id` to the
matching ID in `skeleton.json`, provide the calibrated
`axis_quaternion_wxyz`, and set `calibrated=true` **only after confirming
the iClone FK channel basis**. The template has no approved mappings by design.
Convert with:

```bash
python -m iclone_adapter.anymove_bridge convert \
  runs/dragon/clips/roar/motion.json skeleton.json iclone_fk.json \
  --map-json map.json
```

The output intentionally excludes AnyMove root translation and rotation and
does not compensate for different bone lengths. This is a per-bone rotation
test, **not a finished retarget**. It requires a rig with known correspondence
and verified per-bone axes. To test arbitrary iClone skeletons, first run the
synthetic FK channel diagnostic and confirm axes in the actual iClone UI.
