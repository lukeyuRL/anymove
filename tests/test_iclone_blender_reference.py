"""Blender script optional compare flags; no bpy import in CI."""
import ast
import pathlib
import unittest


SOURCE=pathlib.Path(__file__).resolve().parents[1]/"blender"/"apply_motion.py"


class BlenderReferenceOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text=SOURCE.read_text(encoding="utf-8")
        cls.tree=ast.parse(cls.text,filename=str(SOURCE))

    def test_script_compiles_for_blender_python(self):
        # Syntax check only; Blender's bpy is not installed in ordinary CI.
        compile(self.tree, str(SOURCE), "exec")

    def test_reference_flags_and_keyframe_paths(self):
        literals={node.value for node in ast.walk(self.tree)
                  if isinstance(node,ast.Constant) and isinstance(node.value,str)}
        self.assertIn("--save-blend",literals)
        self.assertIn("--no-root-motion",literals)
        self.assertIn("rotation_euler",literals)
        self.assertIn("rotation_quaternion",literals)
        self.assertIn("location",literals)
        self.assertIn("animation.blend",literals)
        self.assertIn("bpy.ops.wm.save_as_mainfile",self.text)

    def test_no_root_branch_is_explicit_and_preserves_normal_branch(self):
        t=self.text
        self.assertIn('if "--no-root-motion" in opts:',t)
        self.assertIn('Quaternion((1.0, 0.0, 0.0, 0.0))',t)
        self.assertIn('Vector((0.0, 0.0, 0.0))',t)
        self.assertIn('Quaternion(motion["root_quat"][t])',t)
        self.assertIn('Vector(motion["trans"][t])',t)
        self.assertIn('scene.frame_set(0)',t)


if __name__ == "__main__":
    unittest.main()
