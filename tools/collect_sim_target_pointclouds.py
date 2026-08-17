#!/usr/bin/env python3
"""Render wrist-camera simulation point clouds for ICP target generation.

This script reuses the cube pick-place Isaac scene builder so the scene layout,
robot USD, robot base pose, and wrist camera mount match
scense_collect_cube_pick_place_embody_tag.py. It does not run the pick/place
trajectory and does not spawn the cube. Instead, for each real sample it sets
the simulated UR5e arm joints to the recorded real joint values, renders the
wrist camera depth, back-projects it to the RealSense optical frame, and saves
that point cloud in both camera and world coordinates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

COLLECT_SCRIPT = Path(
    "/data/user/isaacsim6/outputs/baselines/broken_precreated_peg_attempt_20260709/scense_collect_cube_pick_place_embody_tag.py"
)
USD_CAMERA_R_OPTICAL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
D405_INTRINSICS = {
    "fx": 394.0833740234375,
    "fy": 393.5317077636719,
    "cx": 318.42926025390625,
    "cy": 235.58648681640625,
}
ALIGN_REAL_BASE_FRAME_T_ISAAC_BASE_LINK_QUAT_WXYZ = (0.0, 0.0, 0.0, 1.0)  # Rz(180deg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real-run-dir", type=Path, default=Path("data/icp_run_20260814_155745"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/isaacsim6_outputs/sim_icp_target_pointclouds"))
    parser.add_argument("--collect-script", type=Path, default=COLLECT_SCRIPT)
    parser.add_argument("--scene-usd", default="/data/user/isaacsim6/assets/810_data/usd-result/embody_tag.usda")
    parser.add_argument("--robot-usd", default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--warmup-steps", type=int, default=72)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--capture-updates", type=int, default=3)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=1.0)
    parser.add_argument("--sample-glob", default="sample_*")
    parser.add_argument("--max-points-per-sample", type=int, default=0, help="0 keeps all depth pixels after filtering.")
    parser.add_argument("--gripper-command", type=float, default=1.0, help="Sim gripper DOF command while rendering target point clouds; 1.0 means closed for this dataset.")
    parser.add_argument("--merged-voxel", type=float, default=0.004)
    parser.add_argument("--save-per-sample-ply", action="store_true")
    parser.add_argument("--hide-gaussians", action="store_true", default=False)
    parser.add_argument(
        "--disable-align-real-base-frame",
        action="store_true",
        help="Do not apply align_real_base_frame_T_isaac_base_link=Rz(180deg) to the simulated robot base.",
    )
    parser.add_argument("--seed", type=int, default=11)
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
        "sim_target_scene.usda",
    ]
    if args.robot_usd:
        argv += ["--robot-usd", str(args.robot_usd)]
    if args.hide_gaussians:
        argv += ["--hide-gaussians"]
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


def apply_align_real_base_frame(module: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Treat collect-script DESIRED_BASE_POSES as world_T_align_real_base_frame."""
    original = tuple(module.DESIRED_BASE_POSES)
    if args.disable_align_real_base_frame:
        return {
            "enabled": False,
            "frame": "isaac_base_link",
            "original_desired_base_poses": original,
            "applied_desired_base_poses": original,
        }

    aligned = []
    for pos, quat in original:
        aligned_quat = module.quat_normalize(
            module.quat_multiply(tuple(float(v) for v in quat), ALIGN_REAL_BASE_FRAME_T_ISAAC_BASE_LINK_QUAT_WXYZ)
        )
        aligned.append((tuple(float(v) for v in pos), aligned_quat))
    module.DESIRED_BASE_POSES = tuple(aligned)
    return {
        "enabled": True,
        "frame": "align_real_base_frame",
        "semantics": (
            "Original DESIRED_BASE_POSES are world_T_align_real_base_frame "
            "(real UR base frame). Applied DESIRED_BASE_POSES are "
            "world_T_isaac_base_link after composing Rz(180deg)."
        ),
        "align_real_base_frame_T_isaac_base_link": {
            "translation": [0.0, 0.0, 0.0],
            "quat_wxyz": list(ALIGN_REAL_BASE_FRAME_T_ISAAC_BASE_LINK_QUAT_WXYZ),
            "matrix": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "original_world_T_align_real_base_frame_poses": original,
        "applied_world_T_isaac_base_link_poses": tuple(aligned),
    }


def read_sample_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_joint_positions(sample_dir: Path) -> np.ndarray:
    metadata = read_sample_metadata(sample_dir)
    joints = np.asarray(metadata["robot"]["joint_position_rad"], dtype=np.float64)
    if joints.shape != (6,):
        raise ValueError(f"{sample_dir}/metadata.json joint_position_rad must be length 6, got {joints.shape}")
    return joints


def set_arm_joints(ctx: Any, module: Any, arm_joints: np.ndarray, gripper_command: float | None = None) -> None:
    dof_positions = ctx.robot.get_dof_positions().numpy().astype(np.float64)
    dof_velocities = ctx.robot.get_dof_velocities().numpy().astype(np.float64)
    dof_positions[:, ctx.arm_dof_indices] = arm_joints.reshape(1, -1)
    dof_velocities[:, ctx.arm_dof_indices] = 0.0
    if gripper_command is not None and ctx.gripper_dof_indices:
        dof_positions[:, ctx.gripper_dof_indices] = float(gripper_command)
        dof_velocities[:, ctx.gripper_dof_indices] = 0.0
    ctx.robot.set_dof_positions(dof_positions)
    ctx.robot.set_dof_velocities(dof_velocities)
    ctx.robot.set_dof_position_targets(dof_positions)
    module.wait_updates(1)
    module.sync_wrist3_usd_to_rigid(ctx, label="sim_icp_target_joint_set")


def gf_matrix_to_numpy(matrix: Any) -> np.ndarray:
    arr = np.eye(4, dtype=np.float64)
    for r in range(4):
        for c in range(4):
            arr[r, c] = float(matrix[r][c])
    # USD Gf.Matrix4d uses row-vector layout; transpose to the
    # column-vector convention used by NumPy helpers in this file.
    return arr.T


def camera_world_t_optical(module: Any, stage: Any, camera_path: str) -> np.ndarray:
    world_t_usd = gf_matrix_to_numpy(module.get_prim_world_matrix(stage, camera_path))
    usd_t_optical = np.eye(4, dtype=np.float64)
    usd_t_optical[:3, :3] = USD_CAMERA_R_OPTICAL
    return world_t_usd @ usd_t_optical


def transform_points(t: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ t[:3, :3].T + t[:3, 3]


def depth_to_points_optical(depth: np.ndarray, intr: dict[str, float], min_depth: float, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    x = (u.astype(np.float64) - float(intr["cx"])) * z / float(intr["fx"])
    y = (v.astype(np.float64) - float(intr["cy"])) * z / float(intr["fy"])
    points = np.stack([x, y, z], axis=1)
    pixels = np.stack([u, v], axis=1).astype(np.uint16)
    return points, pixels


def as_uint8_rgb(rgb: np.ndarray, pixels_uv: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        colors = rgb[pixels_uv[:, 1], pixels_uv[:, 0], :3]
    else:
        colors = np.zeros((len(pixels_uv), 3), dtype=np.uint8)
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors


def random_limit(points: np.ndarray, colors: np.ndarray, pixels: np.ndarray, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if limit <= 0 or len(points) <= limit:
        return points, colors, pixels
    idx = rng.choice(len(points), size=limit, replace=False)
    return points[idx], colors[idx], pixels[idx]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0.0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if colors is None:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8)
    with path.open("wb") as f:
        f.write((
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii"))
        for p, c in zip(points, colors):
            f.write(struct.pack("<fffBBB", float(p[0]), float(p[1]), float(p[2]), int(c[0]), int(c[1]), int(c[2])))


def annotator_data_array(data: Any) -> np.ndarray:
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return np.asarray(data)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = args.output_dir / "debug.txt"
    if debug_path.exists():
        debug_path.unlink()

    module = load_collect_module(args)
    align_real_base_frame = apply_align_real_base_frame(module, args)
    try:
        ctx = module.setup_collect_context(str(args.output_dir), str(debug_path))

        wrist_render_product = module.rep.create.render_product(ctx.wrist_camera_path, (args.width, args.height))
        depth_annotator = module.rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
        rgb_annotator = module.rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        depth_annotator.attach([wrist_render_product])
        rgb_annotator.attach([wrist_render_product])
        module.wait_updates(max(2, args.capture_updates))

        sample_dirs = sorted(p for p in args.real_run_dir.glob(args.sample_glob) if p.is_dir())
        if not sample_dirs:
            raise FileNotFoundError(f"No samples matched {args.real_run_dir / args.sample_glob}")

        merged_points = []
        rows = []
        for sample_dir in sample_dirs:
            sample_name = sample_dir.name
            arm_joints = sample_joint_positions(sample_dir)
            set_arm_joints(ctx, module, arm_joints, gripper_command=args.gripper_command)
            module.wait_updates(max(1, args.settle_steps))
            for _ in range(max(1, args.capture_updates)):
                module.simulation_app.update()

            depth = annotator_data_array(depth_annotator.get_data())
            rgb = annotator_data_array(rgb_annotator.get_data())
            points_cam, pixels_uv = depth_to_points_optical(depth, D405_INTRINSICS, args.min_depth, args.max_depth)
            colors = as_uint8_rgb(rgb, pixels_uv)
            points_cam, colors, pixels_uv = random_limit(points_cam, colors, pixels_uv, args.max_points_per_sample, rng)

            world_t_camera = camera_world_t_optical(module, ctx.stage, ctx.wrist_camera_path)
            points_world = transform_points(world_t_camera, points_cam)
            merged_points.append(points_world)

            out_sample = args.output_dir / sample_name
            out_sample.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_sample / "pointcloud_camera_sim.npz", points_xyz_m=points_cam.astype(np.float32), colors_rgb_uint8=colors, pixels_uv_uint16=pixels_uv)
            np.savez_compressed(out_sample / "pointcloud_world_sim.npz", points_xyz_m=points_world.astype(np.float32), colors_rgb_uint8=colors, pixels_uv_uint16=pixels_uv)
            np.save(out_sample / "world_T_camera_optical.npy", world_t_camera)
            rgb_to_save = rgb[..., :3] if getattr(rgb, "ndim", 0) == 3 and rgb.shape[-1] >= 3 else rgb
            module.imageio.imwrite(out_sample / "rgb_sim.png", np.asarray(rgb_to_save, dtype=np.uint8))
            if args.save_per_sample_ply:
                write_ply(out_sample / "pointcloud_world_sim.ply", points_world, colors)

            rows.append({
                "sample": sample_name,
                "source_sample_dir": str(sample_dir),
                "points": int(len(points_world)),
                "joint_position_rad": arm_joints.tolist(),
                "world_T_camera_optical": world_t_camera.tolist(),
                "camera_path": ctx.wrist_camera_path,
            })
            print(f"{sample_name}: points={len(points_world)}")

        merged = np.concatenate(merged_points, axis=0) if merged_points else np.zeros((0, 3), dtype=np.float64)
        merged_ds = voxel_downsample(merged, args.merged_voxel)
        np.savez_compressed(args.output_dir / "sim_target_world_merged.npz", points_xyz_m=merged_ds.astype(np.float32))
        write_ply(args.output_dir / "sim_target_world_merged.ply", merged_ds)

        metadata = {
            "real_run_dir": str(args.real_run_dir),
            "collect_script": str(args.collect_script),
            "scene_usd": str(module.ARGS.scene_usd),
            "robot_usd": str(module.ARGS.robot_usd),
            "wrist_camera_path": ctx.wrist_camera_path,
            "width": int(args.width),
            "height": int(args.height),
            "intrinsics": D405_INTRINSICS,
            "depth_annotator": "distance_to_image_plane",
            "frame_convention": "pointcloud_camera_sim is RealSense optical: +X right, +Y down, +Z forward; pointcloud_world_sim is USD/ArUco world; rgb_sim.png is the matching simulated wrist RGB frame.",
            "align_real_base_frame": align_real_base_frame,
            "merged_points_before_voxel": int(len(merged)),
            "merged_points_after_voxel": int(len(merged_ds)),
            "merged_voxel_m": float(args.merged_voxel),
            "samples": rows,
        }
        (args.output_dir / "sim_target_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"merged_points={len(merged_ds)}")
        print(f"wrote {args.output_dir / 'sim_target_world_merged.ply'}")
        return 0
    finally:
        module.simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
