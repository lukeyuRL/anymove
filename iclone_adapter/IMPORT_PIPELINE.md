# Dragon reference same-rig Motion Import Pipeline (v0.4.0)

Scope: **same original 42-Bone Dragon rig**, imported into a fresh iClone
scene with the measured **100x FBX parent scale**, running 24 FPS.
This is *not* an arbitrary Avatar retargeter or a production-calibrated
SDK. It formalizes the configuration already visually/numerically tested
against the Blender Dragon roar reference.

## What the pipeline guarantees

The CLI prepares a complete package without running Blender, PyTorch
or RLPy. It locks the source Dragon rest skeleton to the checked-in
`assets/dragon/rigged/rig_info.json` and refuses changed Bone rest data,
wrong Bone order, incomplete snapshots, a foreign hierarchy, wrong 24 FPS,
or a target import scale other than approximately 100x.

For every Bone in every frame (41 nonroot + Pelvis), it checks the target
Rotation XYZ controllers reproduce the expected quaternion after Rest Basis
conjugation, with error <= 0.001 degrees. The Pelvis Position XYZ values
must equal the Blender Pelvis Rest-local conversion of the source `trans`
with the empirically tested multiplier 1.0.

It produces an SHA-256 trace of the source motion, source Bone Rest rig,
target hierarchy and initial world matrices. The manifest hashes the
generated FK file and full-frame mathematical audit. The `verify` command
checks that the bundle is unchanged and still matches source inputs.

**Important limitations:** passing these offline checks does not prove
iClone 8 Keyframe interpolation between frames, dynamic IK evaluation,
skin weighting, independent object transforms or arbitrary rigs. The
saved `calibration_status` remains `unverified_...` so the strict
clean-scene guard still applies to actual iClone writing.

## Prepare a complete package

Run the original AnyMove Solver first to create a valid `motion.json`.
No need to re-run if the motion file already exists.

```powershell
cd F:\anymove
git pull --ff-only origin iClone

py -3.14 -m iclone_adapter.import_pipeline prepare `
  --motion runs\dragon\clips\roar\motion.json `
  --skeleton F:\anymove_test\skeleton.json `
  --output F:\anymove_test\dragon_roar_import
```

Outputs:

- `F:\anymove_test\dragon_roar_import\iclone_fk_full.json` —
  directly load this in the existing iClone Writer
- `F:\anymove_test\dragon_roar_import\offline_frame_audit.json` —
  all-frame, all-Bone quaternion/translation check
- `F:\anymove_test\dragon_roar_import\import_manifest.json` —
  source/basis/profile/hashes/runtime acceptance state

By default the pipeline refuses to overwrite an existing directory.
Use `--overwrite` only to replace its three generated JSON files,
and only if that directory contains no unrelated files.

## Verify package integrity

```powershell
py -3.14 -m iclone_adapter.import_pipeline verify `
  --motion runs\dragon\clips\roar\motion.json `
  --skeleton F:\anymove_test\skeleton.json `
  --output F:\anymove_test\dragon_roar_import
```

## Write animation in iClone

Open a **fresh 24-FPS Dragon project**, select the same unposed Dragon,
move to Frame 0, launch AnyMove v0.4.0 panel, select
`Write FK motion to selected avatar (modifies scene)` and choose
`iclone_fk_full.json`. The existing writer refuses dirty source poses,
prekeyed FK/Position channels and old motion clips; it also supports
FBX-generated static actor initialization Transform keys.

Compare frames 0, 31, 62, 93 and 123 with the Blender
`runs/dragon/clips/roar/full_root_reference/animation.blend`.
The 5-frame Root Translation numerical comparison already passed for the
original reference Roar clip. Other motions using the same Dragon rig
still require visual/live SDK validation.

Do not reuse a scene after writing once. Do not erase actor Transform
initialization keys or manually Reset Pose to bypass the guard.

## Next validation gates

1. Validate **all 124 stored FK/Root Position controller samples** from
   actual iClone SDK readback, rather than only a few frames.
2. Establish subframe interpolation policy and per-frame pose evaluation
   (RINode WorldTransform can be stale during automated timeline scrubbing).
3. Validate at least one additional Dragon clip on the same imported rig.
4. Create a distinct calibration profile for any other rig; do not relax
   the current Dragon reference gate to silently support it.


## v0.4.1 — In-app full-frame FK + Root Position controller readback

The import package now includes a pre-write SHA-256 sidecar check when
opening the generated FK in iClone. The panel blocks a modified/moved
bundled FK file if its adjacent import_manifest.json is missing or
does not match. The CLI verify command still separately rechecks the
original source motion and unposed skeleton against the bundle.

After writing the package on a fresh Dragon project, use the button:

Audit ALL FK and Root Position frames (read-only)

Select the exact same imported iclone_fk_full.json, choose Clip index 0
(unless you explicitly used another), and save runtime_controller_audit.json.

For the 124-frame reference clip, the audit checks all 42 Rotation XYZ
controllers and 1 Pelvis Position XYZ controller:

    124 * (42 + 1) * 3 = 15,996 scalar values.

It requires each axis to contain exactly 124 keys, uses GetValue at
every integer-frame Clip-local time, compares against the JSON with
tolerance 1e-4, and never moves the iClone playhead or writes data.

Expected result:

    status = all_frame_controller_values_match
    verified_scalar_samples = 15996
    mismatched_scalar_samples = 0
    channels_with_wrong_key_count = 0

Any error is reported by Bone, Position/Rotation and first failed frame.
This tests complete **Controller key data** and is distinct from the
offline mathematical fidelity audit. It still cannot prove rendered
pose, IK override or subframe interpolation. Do not call this result
a fully certified animation retarget.


## v0.4.2: Native iClone subframe interpolation survey

The uploaded iClone full-frame report verifies all **15,996 / 15,996**
stored FK and Pelvis Position scalar samples across 124 frames (42
Rotation Bones plus one Pelvis Position). This is the baseline for the
next acceptance gate.

An added, read-only panel button inspects every native iClone curve
at the midpoint of all 123 consecutive frame intervals:

    Survey all FK/Root half-frame interpolations (read-only)

Use the exact same bundled iclone_fk_full.json and Clip 0, then
save subframe_interpolation_audit.json. For a 124-frame clip the
tool reads 123*(42+1)*3 = **15,867 scalar midpoint samples**.

Each RFloatControl.GetValue at a true intermediate RTime tick is compared
with linear interpolation between the bounding integer-frame FK values.
Nonlinearity and overshoot are reported with Bone, channel, axis and
frame pair. Native interpolation differs from linear is NOT automatically
incorrect: Blender's F-Curves can use Bezier or other transition types.
The audit must next be compared with Blender subframes or evaluated Bone
poses before a production interpolation policy is fixed. A failed
GetValue returns subframe_readback_incomplete.

Nothing modifies the Timeline, skeleton or keys. All-frame controller
integrity was previously verified; this survey is a distinct SDK-time
interpolation test, NOT proof of on-screen evaluated bone motion.
