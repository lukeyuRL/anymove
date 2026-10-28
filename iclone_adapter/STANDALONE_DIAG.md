# Standalone P0 diagnostic

When clicking **Export current-pose skeleton snapshot** fails without a debug
file, do not retry the in-app button. This standalone script runs independently
of the existing `iclone_adapter` imports.

## Run

1. Update `iClone` branch and keep iClone open with **one avatar selected**.
2. In iClone select **Script > Load Python** and load the repository-root
   `diag_iclone.py`. No `launch_iclone.py`, UI dialog or manual console code.
3. A message box shows the build `P0-STANDALONE-20261028-01` and actual
   diagnostic filename, normally `%TEMP%\\anymove_iclone_probe.json`.
4. Upload that JSON file. A successful P0 attempt also writes
   `%TEMP%\\anymove_iclone_skeleton.json`.

The first file is created **before** importing RLPy or AnyMove modules. It
captures the exact script location, old loaded module paths, Python executable,
per-stage full Python tracebacks, 103-ish raw Skin Bone names, raw GetID values,
direct parent names/IDs, duplicate groups, and the result of the existing
registry/export routines. Errors in later stages cannot erase earlier captured
diagnostics.

The script only **reads** the iClone scene. It does not create motion clips,
keyframes, or change the actor. The report contains bone names/hierarchy:
review before sharing outside your team.

Test without iClone (expected report mentions missing RLPy):

```bash
python -m unittest discover -s tests -p "test_iclone_*.py" -v
```
