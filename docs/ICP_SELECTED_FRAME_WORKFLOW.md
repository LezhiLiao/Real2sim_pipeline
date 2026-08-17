# Selected-Frame Background ICP Workflow

This workflow is the recommended base-pose refinement path after the 2026-08-17 debugging session.

## Why selected frames

RealSense point clouds from complex background scenes can be noisy even when RGB looks correct. Unconstrained ICP can drift along planes or latch onto local repeated structures, producing large per-frame deltas with deceptively low RMSE. Use only frames whose real/sim RGB and point cloud overlays are visually plausible.

Current selected frames:

```text
sample_000010
sample_000009
sample_000008
sample_000006
sample_000000
```

## Mandatory split

Before ICP, split points by camera optical depth:

```text
background: z >= 0.20 m
near/gripper: z < 0.20 m
```

Use background only for robot base/world ICP. Use gripper/near points only for a separate constrained gripper or hand-eye diagnostic.

## ICP semantics

The implementation uses:

```text
source = real background points in world
target = sim background points in world
```

So the direct ICP result is `delta_real_to_sim`. To correct the simulated robot/world placement into the real domain, use the inverse:

```text
delta_sim_to_real = inverse(delta_real_to_sim)
world_T_align_real_base_new = delta_sim_to_real @ world_T_align_real_base_init
```

Do not apply the delta to a base pose that was not used to generate the sim target/split points. Otherwise the correction is double-counted.

## Capture recommendation

For future calibration datasets, do not use complex cluttered background. Capture:

- Clean tabletop.
- Table front/side edge.
- Stable right-angle geometry if available.
- Low-reflection, non-transparent surfaces.

This gives ICP real 3D constraints without making RealSense depth noise dominate.

## Gripper Near-Point Finetune

For gripper refinement, do not use free 6DoF ICP on `near/gripper` clouds. It gives low RMSE but unstable, physically meaningless deltas.

Use robust center diagnostics in camera optical frame:

```text
near points: 0.05m <= camera_z < 0.20m
delta_camera = real_center - sim_center
```

Current original MAD-filtered center delta:

```text
[-0.0074827425, 0.0007986906, 0.0202399533] m
```

The gripper is connected by a fixed joint, not as a direct child of `wrist_3_link`:

```text
robot_gripper_joint
  body0 = wrist_3_link
  body1 = Robotiq base_link
```

Joint frame semantics:

```text
localPos0/localRot0: joint frame in wrist_3_link coordinates
localPos1/localRot1: joint frame in Robotiq base_link coordinates
world_T_wrist3 @ localFrame0 == world_T_gripper_base @ localFrame1
```

Current +2cm test edits only one side:

```text
localPos0 = (0, 0, 0.02)
localPos1 = (0, 0, 0)
```

Do not move `wrist_3_link`; the wrist camera is mounted there. Do not move only visual meshes, because that can desynchronize visual and collision geometry.
