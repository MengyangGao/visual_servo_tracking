# MuJoCo asset sources

## Built-in robot models

The repository pins Google DeepMind MuJoCo Menagerie as the git submodule `mujoco/vendor/mujoco_menagerie`. Three self-contained model files are registered:

| Robot | MJCF | Asset directory |
| --- | --- | --- |
| Franka Emika Panda (`panda`) | `franka_emika_panda/panda.xml` | `franka_emika_panda/assets/` |
| Universal Robots UR5e (`ur5e`) | `universal_robots_ur5e/ur5e.xml` | `universal_robots_ur5e/assets/` |
| UFactory Lite6 (`lite6`) | `ufactory_lite6/lite6.xml` | `ufactory_lite6/assets/` |

Initialize and inspect the pinned checkout with:

```bash
git submodule update --init --recursive mujoco/vendor/mujoco_menagerie
git submodule status mujoco/vendor/mujoco_menagerie
```

The source-tree default points at that submodule. An installed wheel contains simulator code only, not Menagerie meshes or textures. Set `MUJOCO_MENAGERIE_PATH` to another Menagerie root when using a wheel or an alternate checkout.

Menagerie licensing and attribution are upstream-controlled. Preserve its top-level `LICENSE`, `CITATION.cff`, and model-specific documentation when redistributing any selected robot assets.

## Custom robot assets

`--robot-file` loads a strict descriptor whose `xml_path` and `asset_dir` are resolved relative to the descriptor. The MJCF must be self-contained and cannot use `<include>`. Its controlled joints must be scalar hinge/slide joints with compatible position actuators. The runtime injects the camera, table, visual target, and supporting assets into the parsed MJCF and rewrites compiler mesh/texture directories to absolute paths.

The robot descriptor and all referenced XML, mesh, and texture files remain external runtime assets. They are not copied into the Python package automatically.

## Target assets

Built-in targets are code-defined primitive or compound MJCF geoms in `src/mujoco_servo/targets.py`:

- `apple`
- `bottle`
- `box`
- `capsule`
- `cup`
- `cylinder`
- `dumbbell`
- `hammer`
- `phone`
- `sphere`
- `tower`

Custom target JSON files support:

- MuJoCo `box`, `sphere`, `cylinder`, and `capsule` primitives;
- compound targets assembled from primitive and/or mesh parts;
- external `.obj` and `.stl` meshes with positive XYZ scale.

Mesh paths are resolved relative to the target JSON file and must remain available at runtime. The loader does not download, convert, embed, or copy those files. Users are responsible for asset provenance, compatible units/origin, and redistribution rights.

All target geoms are rendered with collision disabled (`contype=0`, `conaffinity=0`). They are visual-servo references rather than physical graspable objects. A mesh target's required `size` is approximate bounding-box metadata; the actual rendered dimensions come from the mesh coordinates and configured scale.
