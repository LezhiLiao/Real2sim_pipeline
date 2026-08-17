# Experiment Results 2026-08-17

This document records the current quantitative real2sim alignment results for the UR5e + Robotiq + wrist D405 setup. Values are copied from the saved diagnostic outputs under `/tmp/isaacsim6_outputs`.

## Background ICP

Final selected-frame shared background ICP used these frames:

```text
sample_000010, sample_000009, sample_000008, sample_000006, sample_000000
```

The ICP implementation uses real background points as source and sim background points as target, so the direct ICP output is `real->sim`. For correcting sim robot placement into the real domain, use the inverse `sim->real` delta.

### Shared Delta

| Quantity | Value |
|---|---:|
| Selected frames | `000010, 000009, 000008, 000006, 000000` |
| `delta_sim_to_real_translation_m` | `(0.0515144680, -0.1008304344, 0.0255156526)` |
| `delta_sim_to_real_quat_xyzw` | `(-0.0262847960, -0.0196498373, -0.0215452050, 0.9992291016)` |
| Rotation angle | `4.4998 deg` |
| RMSE | `0.011436 m` |
| Correspondence pairs | `3000` |
| Result directory | `/tmp/isaacsim6_outputs/icp_background_shared_delta_selected_frames_10_9_8_6_0_20260817` |

Apply as:

```text
world_T_align_real_base_new = delta_sim_to_real_world @ world_T_align_real_base_init
```

### Per-Frame Free ICP Deltas

These per-frame values are diagnostic only. They show why we do not trust single-frame free ICP as the final base correction.

| Frame | `sim->real` translation m | Rotation deg | RMSE m | Distance to shared m | Rotation to shared deg |
|---|---:|---:|---:|---:|---:|
| `sample_000010` | `(-0.00236, -0.00130, -0.02232)` | `4.21` | `0.004506` | `0.1229` | `2.83` |
| `sample_000009` | `(-0.00866, -0.00158, -0.06830)` | `6.60` | `0.005808` | `0.1492` | `4.58` |
| `sample_000008` | `(-0.01053, -0.00624, -0.13816)` | `8.93` | `0.007793` | `0.1990` | `6.70` |
| `sample_000006` | `(-0.00056, -0.07639, 0.00404)` | `0.50` | `0.008591` | `0.0614` | `4.23` |
| `sample_000000` | `(-0.01497, -0.00014, -0.07529)` | `5.95` | `0.007667` | `0.1572` | `4.57` |

### Per-Frame Spread vs Shared Delta

| Statistic | Value |
|---|---:|
| Translation distance to shared, mean | `0.1379 m` |
| Translation distance to shared, std | `0.0454 m` |
| Translation distance to shared, max | `0.1990 m` |
| Rotation angle to shared, mean | `4.58 deg` |
| Rotation angle to shared, std | `1.24 deg` |
| Rotation angle to shared, max | `6.70 deg` |
| Per-frame RMSE, mean | `0.006873 m` |
| Per-frame RMSE, std | `0.001494 m` |

Interpretation: the background point clouds are good enough for selected-frame shared-delta ICP, but single-frame free ICP remains under-constrained/noisy. Use shared-delta plus visual overlays as the base-pose refinement signal.

## Gripper Near-Point Finetune

Current gripper fixed joint test state:

```text
robot_gripper_joint:
  localPos0 = (0, 0, 0.02)
  localPos1 = (0, 0, 0)
```

Near points are defined in camera optical frame:

```text
0.05 m <= camera_z < 0.20 m
```

Residual direction:

```text
delta = real_robust_center - sim_robust_center
```

### All 11 Frames

| Quantity | x m | y m | z m | Norm m |
|---|---:|---:|---:|---:|
| Mean residual | `-0.00601` | `0.00342` | `0.01575` | `0.01727` |
| Std | `0.00169` | `0.00038` | `0.00078` | `0.00114` |
| Median residual | `-0.00609` | `0.00358` | `0.01549` | - |
| Max norm | - | - | - | `0.02009` |

### Selected Background Frames Only

| Quantity | x m | y m | z m | Norm m |
|---|---:|---:|---:|---:|
| Mean residual | `-0.00567` | `0.00311` | `0.01583` | `0.01721` |
| Std | `0.00218` | `0.00031` | `0.00043` | `0.00097` |

### Selected-Frame Per-Frame Gripper Residuals

| Frame | `real - sim` robust center residual, camera frame m |
|---|---:|
| `sample_000000` | `(-0.00704, 0.00312, 0.01655)` |
| `sample_000006` | `(-0.00179, 0.00263, 0.01526)` |
| `sample_000008` | `(-0.00527, 0.00358, 0.01585)` |
| `sample_000009` | `(-0.00599, 0.00322, 0.01557)` |
| `sample_000010` | `(-0.00823, 0.00297, 0.01591)` |

### Gripper Mount Tests

| Test | Fixed joint edit | Mean robust residual, camera frame m | Output directory |
|---|---|---:|---|
| Original | `localPos0=(0,0,0)`, `localPos1=(0,0,0)` | `(-0.00748, 0.00080, 0.02024)` | original target set |
| Invalid double-sided +1cm | `localPos0=(0,0,0.01)`, `localPos1=(0,0,0.01)` | `(-0.00746, 0.00081, 0.02022)` | `/tmp/isaacsim6_outputs/sim_icp_target_pointclouds_gripper_z_plus_1cm_20260817` |
| Single-sided +1cm | `localPos0=(0,0,0.01)`, `localPos1=(0,0,0)` | `(-0.00716, 0.00207, 0.01900)` | `/tmp/isaacsim6_outputs/sim_icp_target_pointclouds_gripper_joint_localpos0_plus1cm_only_20260817` |
| Current single-sided +2cm | `localPos0=(0,0,0.02)`, `localPos1=(0,0,0)` | `(-0.00601, 0.00342, 0.01575)` | `/tmp/isaacsim6_outputs/sim_icp_target_pointclouds_gripper_joint_localpos0_plus2cm_only_20260817` |

Interpretation: gripper robust center residual is much more stable than free near-point 6DoF ICP. Continue gripper mount tuning with constrained fixed-joint edits and near-point overlays, not free ICP.
