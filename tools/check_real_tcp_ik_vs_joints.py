#!/usr/bin/env python3
"""Compare real recorded joints with Isaac IK solved from real base_T_tcp.

This is a diagnostic for the ICP target rendering path:

1. Set Isaac arm joints from sample_*/metadata.json robot.joint_position_rad.
2. Measure the simulated wrist_3_link pose in robot base frame.
3. Treat the real sample base_T_tcp as if it were wrist_3_link and solve IK.
4. Compare the IK solution with the recorded real joints.

Large FK pose error in step 2 means the saved real TCP frame is probably not
Isaac wrist_3_link, or the joint order/sign convention is wrong. Large joint
error with small FK pose error usually means IK found an equivalent branch.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


COLLECT_SCRIPT = Path(
    "/data/user/isaacsim6/outputs/baselines/broken_precreated_peg_attempt_20260709/scense_collect_cube_pick_place_embody_tag.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real-run-dir", type=Path, default=Path("data/icp_run_20260814_155745"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/isaacsim6_outputs/check_real_tcp_ik_vs_joints"))
    parser.add_argument("--collect-script", type=Path, default=COLLECT_SCRIPT)
    parser.add_argument("--scene-usd", default="/data/user/isaacsim6/assets/810_data/usd-result/embody_tag.usda")
    parser.add_argument("--robot-usd", default="")
    parser.add_argument("--sample-glob", default="sample_*")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--warmup-steps", type=int, default=72)
    parser.add_argument("--capture-updates", type=int, default=3)
    parser.add_argument("--ik-max-iters", type=int, default=120)
    parser.add_argument("--gripper-command", type=float, default=1.0)
    return parser.parse_args()


def load_collect_module(args: argparse.Namespace) -> Any:
    old_argv = sys.argv[:]
    argv = [
        str(args.collect_script),
        "--mode",
        "render",
        "--scene-usd",
        str(args.scene_usd),
        "--output-dir",
        str(args.output_dir),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--renderer",
        str(args.renderer),
        "--warmup-steps",
        str(args.warmup_steps),
        "--capture-updates",
        str(args.capture_updates),
        "--stage-name",
        "check_real_tcp_ik_vs_joints_scene.usda",
    ]
    if args.robot_usd:
        argv += ["--robot-usd", str(args.robot_usd)]
    sys.argv = argv
    try:
        spec = importlib.util.spec_from_file_location("cube_collect_scene", args.collect_script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load collect script: {args.collect_script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["cube_collect_scene"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


def rotmat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_angle_error(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    q1 = q1_wxyz / np.linalg.norm(q1_wxyz)
    q2 = q2_wxyz / np.linalg.norm(q2_wxyz)
    dot = abs(float(np.dot(q1, q2)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def read_sample(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata_path = sample_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    joints = np.asarray(metadata["robot"]["joint_position_rad"], dtype=np.float64)
    if joints.shape != (6,):
        raise ValueError(f"{metadata_path} robot.joint_position_rad must be length 6, got {joints.shape}")
    tcp_path = sample_dir / "base_T_tcp.npy"
    if tcp_path.exists():
        base_t_tcp = np.load(tcp_path).astype(np.float64)
    else:
        base_t_tcp = np.asarray(metadata["robot"]["base_T_tcp"], dtype=np.float64)
    if base_t_tcp.shape != (4, 4):
        raise ValueError(f"{sample_dir} base_T_tcp must be 4x4, got {base_t_tcp.shape}")
    return joints, base_t_tcp


def set_arm_joints(ctx: Any, module: Any, arm_joints: np.ndarray, gripper_command: float) -> None:
    dof_positions = ctx.robot.get_dof_positions().numpy().astype(np.float64)
    dof_velocities = ctx.robot.get_dof_velocities().numpy().astype(np.float64)
    dof_positions[:, ctx.arm_dof_indices] = arm_joints.reshape(1, -1)
    dof_velocities[:, ctx.arm_dof_indices] = 0.0
    if ctx.gripper_dof_indices:
        dof_positions[:, ctx.gripper_dof_indices] = float(gripper_command)
        dof_velocities[:, ctx.gripper_dof_indices] = 0.0
    ctx.robot.set_dof_positions(dof_positions)
    ctx.robot.set_dof_velocities(dof_velocities)
    ctx.robot.set_dof_position_targets(dof_positions)
    module.wait_updates(4)
    module.sync_wrist3_usd_to_rigid(ctx, label="real_joint_fk_check")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load_collect_module(args)
    debug_path = str(args.output_dir / "debug.txt")
    ctx = module.setup_collect_context(str(args.output_dir), debug_path)

    sample_dirs = sorted(p for p in args.real_run_dir.glob(args.sample_glob) if p.is_dir())
    if args.max_samples > 0:
        sample_dirs = sample_dirs[: args.max_samples]
    if not sample_dirs:
        raise FileNotFoundError(f"No samples matched {args.real_run_dir / args.sample_glob}")

    joint_names = list(module.UR5E_ARM_JOINT_NAMES)
    rows: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        recorded_joints, base_t_tcp = read_sample(sample_dir)
        target_pos_base = base_t_tcp[:3, 3].astype(np.float64)
        target_quat_base = rotmat_to_quat_wxyz(base_t_tcp[:3, :3])

        set_arm_joints(ctx, module, recorded_joints, float(args.gripper_command))
        fk_pos_base, fk_quat_base = module.get_wrist3_pose_base(ctx)
        fk_pos_error = fk_pos_base - target_pos_base
        fk_quat_error_rad = quat_angle_error(fk_quat_base, target_quat_base)

        ik_joints = module.solve_wrist3_ik_joint_state(
            ctx,
            target_pos_base,
            target_quat_base,
            max_iters=int(args.ik_max_iters),
        )
        ik_joint_diff = wrap_to_pi(ik_joints - recorded_joints)
        ik_fk_pos_base, ik_fk_quat_base = module.get_wrist3_pose_base(ctx)
        ik_pos_error = ik_fk_pos_base - target_pos_base
        ik_quat_error_rad = quat_angle_error(ik_fk_quat_base, target_quat_base)

        row = {
            "sample": sample_dir.name,
            "joint_names_assumed": joint_names,
            "recorded_joint_position_rad": recorded_joints.tolist(),
            "target_tcp_pos_base": target_pos_base.tolist(),
            "target_tcp_quat_base_wxyz": target_quat_base.tolist(),
            "recorded_joint_fk_wrist3_pos_base": fk_pos_base.tolist(),
            "recorded_joint_fk_wrist3_quat_base_wxyz": fk_quat_base.tolist(),
            "recorded_joint_fk_minus_tcp_pos_m": fk_pos_error.tolist(),
            "recorded_joint_fk_pos_error_norm_m": float(np.linalg.norm(fk_pos_error)),
            "recorded_joint_fk_quat_error_rad": float(fk_quat_error_rad),
            "recorded_joint_fk_quat_error_deg": float(np.degrees(fk_quat_error_rad)),
            "ik_joint_position_rad": ik_joints.tolist(),
            "ik_minus_recorded_joint_wrapped_rad": ik_joint_diff.tolist(),
            "ik_minus_recorded_joint_wrapped_deg": np.degrees(ik_joint_diff).tolist(),
            "ik_joint_error_norm_rad": float(np.linalg.norm(ik_joint_diff)),
            "ik_joint_error_max_abs_rad": float(np.max(np.abs(ik_joint_diff))),
            "ik_fk_minus_tcp_pos_m": ik_pos_error.tolist(),
            "ik_fk_pos_error_norm_m": float(np.linalg.norm(ik_pos_error)),
            "ik_fk_quat_error_rad": float(ik_quat_error_rad),
            "ik_fk_quat_error_deg": float(np.degrees(ik_quat_error_rad)),
        }
        rows.append(row)
        print(
            f"{sample_dir.name}: fk_pos_err={row['recorded_joint_fk_pos_error_norm_m']:.6f}m "
            f"fk_rot_err={row['recorded_joint_fk_quat_error_deg']:.3f}deg "
            f"ik_joint_max={row['ik_joint_error_max_abs_rad']:.6f}rad "
            f"ik_pos_err={row['ik_fk_pos_error_norm_m']:.6f}m"
        )

    summary = {
        "real_run_dir": str(args.real_run_dir),
        "collect_script": str(args.collect_script),
        "scene_usd": str(args.scene_usd),
        "robot_usd": str(args.robot_usd or "collect_script_default"),
        "assumption": "metadata robot.base_T_tcp is treated as Isaac wrist_3_link pose in robot base frame for this diagnostic only.",
        "joint_names_assumed": joint_names,
        "num_samples": len(rows),
        "recorded_joint_fk_pos_error_norm_m_mean": float(np.mean([r["recorded_joint_fk_pos_error_norm_m"] for r in rows])),
        "recorded_joint_fk_pos_error_norm_m_max": float(np.max([r["recorded_joint_fk_pos_error_norm_m"] for r in rows])),
        "recorded_joint_fk_quat_error_deg_mean": float(np.mean([r["recorded_joint_fk_quat_error_deg"] for r in rows])),
        "recorded_joint_fk_quat_error_deg_max": float(np.max([r["recorded_joint_fk_quat_error_deg"] for r in rows])),
        "ik_joint_error_max_abs_rad_mean": float(np.mean([r["ik_joint_error_max_abs_rad"] for r in rows])),
        "ik_joint_error_max_abs_rad_max": float(np.max([r["ik_joint_error_max_abs_rad"] for r in rows])),
        "ik_fk_pos_error_norm_m_mean": float(np.mean([r["ik_fk_pos_error_norm_m"] for r in rows])),
        "ik_fk_pos_error_norm_m_max": float(np.max([r["ik_fk_pos_error_norm_m"] for r in rows])),
        "samples": rows,
    }
    json_path = args.output_dir / "tcp_ik_vs_joints_result.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "tcp_ik_vs_joints_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "recorded_joint_fk_pos_error_norm_m",
                "recorded_joint_fk_quat_error_deg",
                "ik_joint_error_max_abs_rad",
                "ik_joint_error_norm_rad",
                "ik_fk_pos_error_norm_m",
                "ik_fk_quat_error_deg",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    module.simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
