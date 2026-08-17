#!/usr/bin/env python3
"""Depth-split ICP: background robot-base delta plus 1-DOF gripper z delta.

The split is done in each camera optical frame:
  background: z >= --split-depth
  near/gripper: z < --split-depth

Background ICP estimates a shared left-multiplied world correction for
world_T_align_real_base_frame. Gripper refinement then keeps that base fixed and
optimizes one scalar dz applied to simulated near points along each frame's
wrist_3_link local +Z direction in world coordinates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from icp_refine_robot_world_pose import (
    USD_CAMERA_R_OPTICAL,
    clean_points,
    default_wrist3_t_camera_optical,
    load_transform,
    make_transform,
    matrix_to_quat_xyzw,
    nearest_neighbors_bruteforce,
    quat_xyzw_to_matrix,
    random_limit,
    read_npz_cloud,
    remove_world_boxes,
    rigid_transform_svd,
    transform_points,
    voxel_downsample,
    write_ply,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--real-run-dir", type=Path, default=Path("data/icp_run_20260814_155745"))
    p.add_argument("--sim-target-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/isaacsim6_outputs/icp_background_and_gripper_z"))
    p.add_argument("--world-T-align-real-base-init", type=Path, default=None)
    p.add_argument("--wrist3-T-camera", type=Path, default=None)
    p.add_argument("--sample-glob", default="sample_*")
    p.add_argument("--split-depth", type=float, default=0.20)
    p.add_argument("--min-depth", type=float, default=0.05)
    p.add_argument("--max-depth", type=float, default=1.0)
    p.add_argument("--source-voxel", type=float, default=0.006)
    p.add_argument("--target-voxel", type=float, default=0.006)
    p.add_argument("--near-source-voxel", type=float, default=0.003)
    p.add_argument("--near-target-voxel", type=float, default=0.003)
    p.add_argument("--max-source-points-per-frame", type=int, default=2500)
    p.add_argument("--max-target-points-per-frame", type=int, default=6000)
    p.add_argument("--max-near-source-points-per-frame", type=int, default=3500)
    p.add_argument("--max-near-target-points-per-frame", type=int, default=5000)
    p.add_argument("--max-pairs-per-frame", type=int, default=600)
    p.add_argument("--max-correspondence-distance", type=float, default=0.05)
    p.add_argument("--near-max-correspondence-distance", type=float, default=0.025)
    p.add_argument("--trim-fraction", type=float, default=0.70)
    p.add_argument("--near-trim-fraction", type=float, default=0.70)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--gripper-z-search-min", type=float, default=-0.05)
    p.add_argument("--gripper-z-search-max", type=float, default=0.05)
    p.add_argument("--gripper-z-coarse-step", type=float, default=0.002)
    p.add_argument("--gripper-z-fine-step", type=float, default=0.00025)
    p.add_argument("--exclude-world-box", type=float, nargs=6, action="append", default=[])
    p.add_argument("--write-clouds", action="store_true")
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def world_t_base_from_sim_metadata(sim_target_dir: Path) -> np.ndarray:
    meta_path = sim_target_dir / "sim_target_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No {meta_path}; pass --world-T-align-real-base-init explicitly")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    align = meta.get("align_real_base_frame") or {}
    poses = align.get("original_world_T_align_real_base_frame_poses")
    if not poses:
        raise ValueError(f"{meta_path} does not contain original_world_T_align_real_base_frame_poses")
    pos, quat_xyzw = poses[0]
    return make_transform(quat_xyzw_to_matrix(quat_xyzw), pos)


def split_by_depth(points: np.ndarray, split_depth: float, min_depth: float, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    points = clean_points(points)
    z = points[:, 2]
    valid = (z >= min_depth) & (z <= max_depth)
    points = points[valid]
    z = points[:, 2]
    near = points[z < split_depth]
    background = points[z >= split_depth]
    return near, background


def select_colors(colors: np.ndarray | None, mask_idx: np.ndarray | None) -> np.ndarray | None:
    if colors is None or mask_idx is None:
        return None
    return colors[mask_idx]


def split_npz_arrays(path: Path, split_depth: float, min_depth: float, max_depth: float) -> dict[str, Any]:
    data = np.load(path)
    points = np.asarray(data["points_xyz_m"], dtype=np.float64)
    colors = np.asarray(data["colors_rgb_uint8"], dtype=np.uint8) if "colors_rgb_uint8" in data else None
    pixels = np.asarray(data["pixels_uv_uint16"], dtype=np.uint16) if "pixels_uv_uint16" in data else None
    finite = np.isfinite(points).all(axis=1)
    z = points[:, 2]
    valid = finite & (z >= min_depth) & (z <= max_depth)
    near_mask = valid & (z < split_depth)
    bg_mask = valid & (z >= split_depth)
    def pack(mask: np.ndarray) -> dict[str, Any]:
        out = {"points": points[mask]}
        if colors is not None:
            out["colors"] = colors[mask]
        if pixels is not None:
            out["pixels"] = pixels[mask]
        return out
    return {"near": pack(near_mask), "background": pack(bg_mask)}


def write_npz(path: Path, points: np.ndarray, colors: np.ndarray | None = None, pixels: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"points_xyz_m": points.astype(np.float32)}
    if colors is not None:
        payload["colors_rgb_uint8"] = colors.astype(np.uint8)
    if pixels is not None:
        payload["pixels_uv_uint16"] = pixels.astype(np.uint16)
    np.savez_compressed(path, **payload)


def load_frames(args: argparse.Namespace, world_t_base: np.ndarray, wrist3_t_camera: np.ndarray, rng: np.random.Generator):
    frames = []
    split_dir = args.out_dir / "depth_split_clouds"
    for real_sample in sorted(p for p in args.real_run_dir.glob(args.sample_glob) if p.is_dir()):
        name = real_sample.name
        sim_sample = args.sim_target_dir / name
        real_cloud_path = real_sample / "pointcloud_camera.npz"
        fk_path = real_sample / "base_T_tcp.npy"
        sim_cam_path = sim_sample / "pointcloud_camera_sim.npz"
        if not (real_cloud_path.exists() and fk_path.exists() and sim_cam_path.exists()):
            continue

        real_split = split_npz_arrays(real_cloud_path, args.split_depth, args.min_depth, args.max_depth)
        sim_cam_split = split_npz_arrays(sim_cam_path, args.split_depth, args.min_depth, args.max_depth)

        base_t_tcp = np.load(fk_path).astype(np.float64)
        world_t_camera = world_t_base @ base_t_tcp @ wrist3_t_camera
        real_near_world = transform_points(world_t_camera, real_split["near"]["points"])
        real_bg_world = transform_points(world_t_camera, real_split["background"]["points"])
        sim_near_world = transform_points(world_t_camera, sim_cam_split["near"]["points"])
        sim_bg_world = transform_points(world_t_camera, sim_cam_split["background"]["points"])
        real_near_world = remove_world_boxes(real_near_world, args.exclude_world_box)
        real_bg_world = remove_world_boxes(real_bg_world, args.exclude_world_box)
        sim_near_world = remove_world_boxes(sim_near_world, args.exclude_world_box)
        sim_bg_world = remove_world_boxes(sim_bg_world, args.exclude_world_box)

        out_sample = split_dir / name
        write_npz(out_sample / "real_near_lt_20cm_world.npz", real_near_world, real_split["near"].get("colors"), real_split["near"].get("pixels"))
        write_npz(out_sample / "real_background_ge_20cm_world.npz", real_bg_world, real_split["background"].get("colors"), real_split["background"].get("pixels"))
        write_npz(out_sample / "sim_near_lt_20cm_world.npz", sim_near_world, sim_cam_split["near"].get("colors"), sim_cam_split["near"].get("pixels"))
        write_npz(out_sample / "sim_background_ge_20cm_world.npz", sim_bg_world, sim_cam_split["background"].get("colors"), sim_cam_split["background"].get("pixels"))
        if args.write_clouds:
            write_ply(out_sample / "real_background_ge_20cm_world.ply", real_bg_world, (255, 80, 60))
            write_ply(out_sample / "sim_background_ge_20cm_world.ply", sim_bg_world, (180, 180, 180))
            write_ply(out_sample / "real_near_lt_20cm_world.ply", real_near_world, (255, 160, 40))
            write_ply(out_sample / "sim_near_lt_20cm_world.ply", sim_near_world, (80, 220, 255))

        real_bg_icp = random_limit(voxel_downsample(real_bg_world, args.source_voxel), args.max_source_points_per_frame, rng)
        sim_bg_icp = random_limit(voxel_downsample(sim_bg_world, args.target_voxel), args.max_target_points_per_frame, rng)
        real_near_icp = random_limit(voxel_downsample(real_near_world, args.near_source_voxel), args.max_near_source_points_per_frame, rng)
        sim_near_icp = random_limit(voxel_downsample(sim_near_world, args.near_target_voxel), args.max_near_target_points_per_frame, rng)
        world_t_wrist3 = world_t_base @ base_t_tcp
        wrist_z_world = world_t_wrist3[:3, 2].astype(np.float64)
        wrist_z_world /= np.linalg.norm(wrist_z_world)

        if len(real_bg_icp) >= 6 and len(sim_bg_icp) >= 6:
            frames.append({
                "name": name,
                "background_source": real_bg_icp,
                "background_target": sim_bg_icp,
                "near_source": real_near_icp,
                "near_target": sim_near_icp,
                "wrist_z_world": wrist_z_world,
                "counts": {
                    "real_near": int(len(real_near_world)),
                    "real_background": int(len(real_bg_world)),
                    "sim_near": int(len(sim_near_world)),
                    "sim_background": int(len(sim_bg_world)),
                },
            })
    if not frames:
        raise FileNotFoundError("No usable matched frames after depth split")
    return frames


def shared_background_icp(frames: list[dict[str, Any]], args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]]]:
    delta = np.eye(4, dtype=np.float64)
    history = []
    max_dist2 = args.max_correspondence_distance ** 2
    trim = min(max(args.trim_fraction, 0.05), 1.0)
    prev = None
    for it in range(args.iterations):
        all_src, all_dst, stats = [], [], []
        for f in frames:
            moved = transform_points(delta, f["background_source"])
            idx, dist2 = nearest_neighbors_bruteforce(moved, f["background_target"])
            valid_idx = np.flatnonzero(dist2 <= max_dist2)
            if len(valid_idx) < 6:
                stats.append({"name": f["name"], "pairs": 0, "rmse_m": None})
                continue
            order = valid_idx[np.argsort(dist2[valid_idx])]
            keep = order[: max(6, int(math.ceil(len(order) * trim)))]
            if args.max_pairs_per_frame > 0 and len(keep) > args.max_pairs_per_frame:
                keep = keep[np.linspace(0, len(keep) - 1, args.max_pairs_per_frame).astype(np.int64)]
            all_src.append(moved[keep])
            all_dst.append(f["background_target"][idx[keep]])
            stats.append({"name": f["name"], "pairs": int(len(keep)), "rmse_m": float(np.sqrt(np.mean(dist2[keep]))), "fitness": float(len(keep)/max(1,len(f["background_source"])))})
        if not all_src:
            raise RuntimeError(f"No background correspondences at iteration {it+1}")
        src = np.concatenate(all_src, axis=0)
        dst = np.concatenate(all_dst, axis=0)
        step = rigid_transform_svd(src, dst)
        delta = step @ delta
        rmse = float(np.sqrt(np.mean(np.sum((transform_points(step, src) - dst) ** 2, axis=1))))
        history.append({"iteration": it + 1, "rmse_m": rmse, "pairs": int(len(src)), "frames": stats})
        if prev is not None and abs(prev - rmse) < 1e-5:
            break
        prev = rmse
    return delta, history


def gripper_z_loss(frames: list[dict[str, Any]], delta_world: np.ndarray, dz: float, args: argparse.Namespace) -> dict[str, Any] | None:
    max_dist2 = args.near_max_correspondence_distance ** 2
    trim = min(max(args.near_trim_fraction, 0.05), 1.0)
    sqs = []
    pairs_total = 0
    frame_stats = []
    for f in frames:
        if len(f["near_source"]) < 6 or len(f["near_target"]) < 6:
            frame_stats.append({"name": f["name"], "pairs": 0, "rmse_m": None})
            continue
        src = transform_points(delta_world, f["near_source"])
        target = f["near_target"] + f["wrist_z_world"].reshape(1, 3) * float(dz)
        idx, dist2 = nearest_neighbors_bruteforce(src, target)
        valid_idx = np.flatnonzero(dist2 <= max_dist2)
        if len(valid_idx) < 6:
            frame_stats.append({"name": f["name"], "pairs": 0, "rmse_m": None})
            continue
        order = valid_idx[np.argsort(dist2[valid_idx])]
        keep = order[: max(6, int(math.ceil(len(order) * trim)))]
        if args.max_pairs_per_frame > 0 and len(keep) > args.max_pairs_per_frame:
            keep = keep[np.linspace(0, len(keep) - 1, args.max_pairs_per_frame).astype(np.int64)]
        sqs.append(dist2[keep])
        pairs_total += len(keep)
        frame_stats.append({"name": f["name"], "pairs": int(len(keep)), "rmse_m": float(np.sqrt(np.mean(dist2[keep]))), "fitness": float(len(keep)/max(1,len(f["near_source"])))})
    if not sqs:
        return None
    all_sq = np.concatenate(sqs)
    return {"dz_m": float(dz), "rmse_m": float(np.sqrt(np.mean(all_sq))), "pairs": int(pairs_total), "frames": frame_stats}


def optimize_gripper_z(frames: list[dict[str, Any]], delta_world: np.ndarray, args: argparse.Namespace) -> tuple[float, list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    coarse = np.arange(args.gripper_z_search_min, args.gripper_z_search_max + args.gripper_z_coarse_step * 0.5, args.gripper_z_coarse_step)
    best = None
    for dz in coarse:
        res = gripper_z_loss(frames, delta_world, float(dz), args)
        if res is None:
            continue
        res["stage"] = "coarse"
        history.append(res)
        if best is None or res["rmse_m"] < best["rmse_m"]:
            best = res
    if best is None:
        raise RuntimeError("No valid near/gripper correspondences during coarse z search")
    lo = max(args.gripper_z_search_min, best["dz_m"] - args.gripper_z_coarse_step)
    hi = min(args.gripper_z_search_max, best["dz_m"] + args.gripper_z_coarse_step)
    fine = np.arange(lo, hi + args.gripper_z_fine_step * 0.5, args.gripper_z_fine_step)
    for dz in fine:
        res = gripper_z_loss(frames, delta_world, float(dz), args)
        if res is None:
            continue
        res["stage"] = "fine"
        history.append(res)
        if res["rmse_m"] < best["rmse_m"]:
            best = res
    return float(best["dz_m"]), history


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    world_t_base = load_transform(args.world_T_align_real_base_init) if args.world_T_align_real_base_init else world_t_base_from_sim_metadata(args.sim_target_dir)
    wrist3_t_camera = load_transform(args.wrist3_T_camera) if args.wrist3_T_camera else default_wrist3_t_camera_optical()

    frames = load_frames(args, world_t_base, wrist3_t_camera, rng)
    delta_world, bg_history = shared_background_icp(frames, args)
    refined_world_t_base = delta_world @ world_t_base
    dz, z_history = optimize_gripper_z(frames, delta_world, args)

    np.save(args.out_dir / "delta_world_T_world_init_background.npy", delta_world)
    np.save(args.out_dir / "world_T_align_real_base_init.npy", world_t_base)
    np.save(args.out_dir / "world_T_align_real_base_refined_background.npy", refined_world_t_base)
    np.save(args.out_dir / "wrist3_T_camera_optical.npy", wrist3_t_camera)

    if args.write_clouds:
        bg_src = np.concatenate([f["background_source"] for f in frames], axis=0)
        bg_src_ref = transform_points(delta_world, bg_src)
        bg_tgt = np.concatenate([f["background_target"] for f in frames], axis=0)
        near_src = np.concatenate([f["near_source"] for f in frames if len(f["near_source"])], axis=0)
        near_src_ref = transform_points(delta_world, near_src)
        near_tgt_adj = np.concatenate([f["near_target"] + f["wrist_z_world"].reshape(1,3)*dz for f in frames if len(f["near_target"])], axis=0)
        write_ply(args.out_dir / "background_source_initial_world_concat.ply", bg_src, (255, 80, 60))
        write_ply(args.out_dir / "background_source_refined_world_concat.ply", bg_src_ref, (70, 180, 255))
        write_ply(args.out_dir / "background_target_world_concat.ply", bg_tgt, (180, 180, 180))
        write_ply(args.out_dir / "near_source_refined_world_concat.ply", near_src_ref, (255, 160, 40))
        write_ply(args.out_dir / "near_target_gripper_z_adjusted_world_concat.ply", near_tgt_adj, (80, 220, 255))

    best_z = min(z_history, key=lambda r: r["rmse_m"])
    result = {
        "real_run_dir": str(args.real_run_dir),
        "sim_target_dir": str(args.sim_target_dir),
        "split_depth_m": float(args.split_depth),
        "frame_count": len(frames),
        "frame_counts": {f["name"]: f["counts"] for f in frames},
        "world_T_align_real_base_init": world_t_base.tolist(),
        "delta_world_T_world_init_background": delta_world.tolist(),
        "delta_background_translation_m": delta_world[:3, 3].tolist(),
        "delta_background_quat_xyzw": matrix_to_quat_xyzw(delta_world[:3, :3]).tolist(),
        "world_T_align_real_base_refined_background": refined_world_t_base.tolist(),
        "world_T_align_real_base_refined_background_translation_m": refined_world_t_base[:3, 3].tolist(),
        "world_T_align_real_base_refined_background_quat_xyzw": matrix_to_quat_xyzw(refined_world_t_base[:3, :3]).tolist(),
        "gripper_mount_refinement": {
            "optimized_parameter": "delta_z only, in wrist_3_link local +Z, applied to simulated gripper/near target points",
            "delta_x_m": 0.0,
            "delta_y_m": 0.0,
            "delta_z_m": dz,
            "best_rmse_m": best_z["rmse_m"],
            "best_pairs": best_z["pairs"],
            "sign_convention": "p_sim_gripper_adjusted_world = p_sim_gripper_world + world_R_wrist3[:,2] * delta_z_m",
        },
        "background_icp_history": bg_history,
        "gripper_z_search_history": z_history,
        "notes": [
            "Depth split is performed in each camera optical frame before world transform.",
            "Background uses points with camera z >= split_depth_m.",
            "Near/gripper uses points with camera z < split_depth_m.",
            "Sim world split clouds are recomputed from pointcloud_camera_sim.npz using real FK + handeye; saved pointcloud_world_sim.npz is not trusted for ICP here.",
            "This script estimates gripper delta_z but does not write it into the robot USD; inspect the gripper joint/root before applying.",
        ],
    }
    (args.out_dir / "background_icp_and_gripper_z_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    last_bg = bg_history[-1]
    print(f"frames={len(frames)} background_pairs={last_bg['pairs']} background_rmse_m={last_bg['rmse_m']:.6f}")
    print(f"delta_background_translation_m={result['delta_background_translation_m']}")
    print(f"world_T_align_real_base_refined_translation_m={result['world_T_align_real_base_refined_background_translation_m']}")
    print(f"gripper_delta_z_m={dz:.6f} near_rmse_m={best_z['rmse_m']:.6f} pairs={best_z['pairs']}")
    print(f"wrote {args.out_dir / 'background_icp_and_gripper_z_result.json'}")


if __name__ == "__main__":
    main()
