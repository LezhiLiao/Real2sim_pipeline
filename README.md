# Real2Sim Pipeline: ArUco/AprilTag 3DGS + Isaac UR5e ICP

This repository packages the current real2sim workflow for a UR5e + Robotiq 2F85 + wrist RealSense D405 setup. It combines:

- ArUco/AprilTag-defined world coordinates.
- 3DGS background reconstruction aligned to the marker/world frame, using the approach from `apriltag-3dgs-align`.
- Isaac Sim target rendering with the same scene layout, robot asset, wrist camera mount, and recorded real joints.
- Background-only ICP to refine the robot base pose in the world frame.
- Separate gripper/near-point refinement so gripper geometry does not corrupt base alignment.

Upstream 3DGS/tag alignment reference:

```text
https://github.com/LezhiLiao/apriltag-3dgs-align
```

## Coordinate Frames

The global `world` frame is the ArUco/AprilTag frame. The real robot base is named `align_real_base_frame` to keep it separate from Isaac's asset `base_link`.

```text
world / ArUco
  -> align_real_base_frame      # real UR base frame
  -> isaac_base_link            # Isaac asset base_link, Rz(180deg) from real base
  -> wrist_3_link
  -> wrist_3_link/realsense     # direct Camera prim, Isaac -Z convention
```

The fixed base-frame relation is:

```text
align_real_base_frame_T_isaac_base_link = Rz(180deg)
quat_wxyz = (0, 0, 0, 1)
```

The wrist camera is mounted directly under `wrist_3_link`. Do not use `ft_frame` and do not use the internal transform of an `rsd455.usd` payload.

## Full Pipeline

1. Define the world frame with an ArUco/AprilTag marker.
2. Capture and reconstruct the static background with 3DGS.
3. Use `apriltag-3dgs-align` style tag constraints to align the 3DGS scene into the world frame.
4. Calibrate real camera intrinsics, fixed camera extrinsics, and wrist hand-eye.
5. Convert wrist hand-eye from OpenCV optical convention to Isaac Camera `-Z` convention before writing the USD Camera prim.
6. Calibrate `world_T_align_real_base_frame` from the real robot to the marker/world frame.
7. Generate sim target point clouds by setting Isaac joints from real `joint_position_rad` and rendering wrist RGB/depth.
8. Visualize real/sim RGB and point cloud overlays before ICP.
9. Split point clouds into background and near/gripper sets.
10. Run selected-frame shared-delta background ICP.
11. Apply `delta_sim_to_real_world` to `world_T_align_real_base_frame` by left multiplication.
12. Separately refine gripper/near geometry if needed.
13. Reconstruct foreground assets such as peg, hole, and cube separately, with simple collision geometry.

## ICP Rules That Matter

Background and gripper point clouds must be separated before optimization:

```text
background: camera optical z >= 0.20 m
near/gripper: camera optical z < 0.20 m
```

Use background for robot base/world refinement. Use near/gripper points only in a separate constrained stage. Mixing them lets close gripper points dominate the objective and turns base ICP into a hand-eye/gripper-error fit.

For calibration captures, prefer low-noise geometry:

- Clean tabletop.
- Clear table edges.
- One additional right-angle or vertical structure if available.
- Avoid complex cluttered backgrounds, reflective surfaces, transparent objects, thin structures, and far RealSense depth.

Always visualize first:

- Compare real `rgb.png` and sim `rgb_sim.png`.
- Overlay real/sim background point clouds in the same world projection.
- Select only frames whose image content and point cloud overlays are plausible.

Current selected frames from the existing dataset:

```text
sample_000010
sample_000009
sample_000008
sample_000006
sample_000000
```

## Current Validated Values

Current wrist hand-eye matrix in OpenCV optical convention:

```text
[  0.9976653405   -0.009330144963   0.0676521745   -0.01836250805 ]
[ -0.02127699195   0.8988521473     0.4377352018   -0.0800952162  ]
[ -0.06489343521  -0.4381526739     0.8965551162    0.07073644041 ]
[  0                0                0               1             ]
```

Written to Isaac USD Camera after:

```text
R_usd = R_handeye @ diag(1, -1, -1)
```

Current cube `world_T_align_real_base_frame` after selected-frame ICP:

```text
translation = (-0.6062680991476823, -0.7083919554004651, 0.1128684852177913)
quat_wxyz   = (0.9993668817805874, -0.02892215679558249, 0.015075419773920105, -0.01421534962909488)
```

Selected-frame shared ICP result:

```text
delta_sim_to_real_translation_m = (0.051514468014859295, -0.10083043438638568, 0.02551565264491571)
delta_sim_to_real_quat_xyzw     = (-0.026284795973907347, -0.019649837311149106, -0.021545205017915146, 0.9992291016253437)
rmse_m = 0.011436
```

Apply it as:

```text
world_T_align_real_base_new = delta_sim_to_real_world @ world_T_align_real_base_init
```

## Gripper Finetune

Do not run unconstrained 6DoF ICP on near/gripper points. The near gripper point clouds are internally stable within real and sim sets, but sim-vs-real free ICP can drift because the geometry is local, partly symmetric, and RealSense near-depth noise changes point density.

Recommended flow:

1. Use near points in camera optical frame: `0.05m <= z < 0.20m`.
2. Compute robust centers for real and sim with MAD filtering or 10%-90% trimmed mean.
3. Use `delta_camera = real_center - sim_center` as a diagnostic.
4. If testing gripper mount changes, constrain the edit to one fixed-joint degree/axis.
5. Re-render sim target point clouds and inspect real/sim near overlays.

Current Robotiq mount structure:

```text
robot_gripper_joint
  body0 = /Root/ur5e/ur5e/wrist_3_link
  body1 = /Root/Robotiq_2F_85_edit/Robotiq_2F_85_edit/Robotiq_2F_85/base_link
```

`localPos0` is the joint frame in wrist_3_link coordinates. `localPos1` is the joint frame in Robotiq base_link coordinates. To change relative mount position, edit one side only. Do not move `wrist_3_link` itself, because the wrist camera is attached to it.

Current tested USD state:

```text
physics:localPos0 = (0, 0, 0.02)
physics:localPos1 = (0, 0, 0)
```

+2cm output:

```text
/tmp/isaacsim6_outputs/sim_icp_target_pointclouds_gripper_joint_localpos0_plus2cm_only_20260817
```

## Tools

```text
tools/collect_sim_target_pointclouds.py       # render sim wrist RGB/depth/point clouds from real joint samples
tools/icp_background_and_gripper_z.py         # split background/near and run base + gripper refinement
tools/icp_refine_robot_world_pose.py          # shared utilities and legacy merged ICP helpers
tools/icp_refine_robot_world_pose_per_frame.py# per-frame diagnostics
tools/check_real_tcp_ik_vs_joints.py          # FK/IK base-frame diagnostic
tools/capture_initial_tcp_pose_png.py         # camera pose debug capture
```

## Documentation

The detailed current setup notes are in:

```text
docs/real2sim_current_setup_notes.txt
docs/EXPERIMENT_RESULTS_20260817.md
```

## Isaac Command Convention

Run Isaac scripts with cleared Isaac/conda environment variables:

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH -u CARB_APP_PATH -u EXP_PATH -u ISAAC_PATH   -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER   TMPDIR=/tmp/isaacsim6_tmp   /tmp/isaacsim6_clean_local/python.sh   tools/collect_sim_target_pointclouds.py     --real-run-dir /data/user/isaacsim6/Project/icp_verify/data/icp_run_20260814_155745     --output-dir /tmp/isaacsim6_outputs/sim_icp_target_pointclouds_RUN     --save-per-sample-ply     --gripper-command 1.0
```
