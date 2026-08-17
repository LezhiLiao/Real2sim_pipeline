#!/usr/bin/env python3
"""Per-frame shared-delta ICP for robot world pose refinement."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from icp_refine_robot_world_pose import (
    clean_points,
    default_world_t_base,
    default_wrist3_t_camera_optical,
    load_transform,
    matrix_to_quat_xyzw,
    nearest_neighbors_bruteforce,
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
    p.add_argument("--sim-target-dir", type=Path, required=True, help="Directory containing sample_*/pointcloud_world_sim.npz")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/isaacsim6_outputs/icp_refine_per_frame"))
    p.add_argument("--world-T-base-init", type=Path, default=None)
    p.add_argument("--wrist3-T-camera", type=Path, default=None)
    p.add_argument("--sample-glob", default="sample_*")
    p.add_argument("--min-depth", type=float, default=0.05)
    p.add_argument("--max-depth", type=float, default=1.0)
    p.add_argument("--source-voxel", type=float, default=0.006)
    p.add_argument("--target-voxel", type=float, default=0.006)
    p.add_argument("--max-source-points-per-frame", type=int, default=2500)
    p.add_argument("--max-target-points-per-frame", type=int, default=6000)
    p.add_argument("--max-pairs-per-frame", type=int, default=600)
    p.add_argument("--max-correspondence-distance", type=float, default=0.06)
    p.add_argument("--trim-fraction", type=float, default=0.70)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--exclude-world-box", type=float, nargs=6, action="append", default=[])
    p.add_argument("--write-clouds", action="store_true")
    p.add_argument("--seed", type=int, default=19)
    return p.parse_args()


def load_frame_clouds(args: argparse.Namespace, world_t_base: np.ndarray, wrist3_t_camera: np.ndarray, rng: np.random.Generator):
    frames = []
    for real_sample in sorted(p for p in args.real_run_dir.glob(args.sample_glob) if p.is_dir()):
        name = real_sample.name
        target_path = args.sim_target_dir / name / "pointcloud_world_sim.npz"
        real_path = real_sample / "pointcloud_camera.npz"
        fk_path = real_sample / "base_T_tcp.npy"
        if not target_path.exists() or not real_path.exists() or not fk_path.exists():
            continue

        points_cam, _ = read_npz_cloud(real_path)
        points_cam = clean_points(points_cam)
        depth = points_cam[:, 2]
        points_cam = points_cam[(depth >= args.min_depth) & (depth <= args.max_depth)]
        points_cam = random_limit(points_cam, args.max_source_points_per_frame * 4, rng)
        base_t_tcp = np.load(fk_path).astype(np.float64)
        source_base = transform_points(base_t_tcp @ wrist3_t_camera, points_cam)
        source_world = transform_points(world_t_base, source_base)
        source_world = remove_world_boxes(source_world, args.exclude_world_box)
        source_world = voxel_downsample(source_world, args.source_voxel)
        source_world = random_limit(source_world, args.max_source_points_per_frame, rng)

        target_world, _ = read_npz_cloud(target_path)
        target_world = clean_points(target_world)
        target_world = remove_world_boxes(target_world, args.exclude_world_box)
        target_world = voxel_downsample(target_world, args.target_voxel)
        target_world = random_limit(target_world, args.max_target_points_per_frame, rng)

        if len(source_world) >= 6 and len(target_world) >= 6:
            frames.append({"name": name, "source": source_world, "target": target_world})
    if not frames:
        raise FileNotFoundError("No usable matched real/sim sample frames found")
    return frames


def shared_delta_icp(frames, args: argparse.Namespace, rng: np.random.Generator):
    delta = np.eye(4, dtype=np.float64)
    history = []
    max_dist2 = float(args.max_correspondence_distance) ** 2
    trim_fraction = min(max(float(args.trim_fraction), 0.05), 1.0)
    prev_rmse = None

    for iteration in range(int(args.iterations)):
        all_src = []
        all_dst = []
        frame_stats = []
        for frame in frames:
            moved = transform_points(delta, frame["source"])
            idx, dist2 = nearest_neighbors_bruteforce(moved, frame["target"])
            valid = dist2 <= max_dist2
            valid_idx = np.flatnonzero(valid)
            if len(valid_idx) < 6:
                frame_stats.append({"name": frame["name"], "pairs": 0, "rmse_m": None})
                continue
            order = valid_idx[np.argsort(dist2[valid])]
            keep_n = max(6, int(math.ceil(len(order) * trim_fraction)))
            keep = order[:keep_n]
            if args.max_pairs_per_frame > 0 and len(keep) > args.max_pairs_per_frame:
                keep = keep[np.linspace(0, len(keep) - 1, args.max_pairs_per_frame).astype(np.int64)]
            all_src.append(moved[keep])
            all_dst.append(frame["target"][idx[keep]])
            frame_stats.append({
                "name": frame["name"],
                "pairs": int(len(keep)),
                "rmse_m": float(np.sqrt(np.mean(dist2[keep]))),
                "fitness": float(len(keep) / max(1, len(frame["source"]))),
            })
        if not all_src:
            raise RuntimeError(f"No valid correspondences at iteration {iteration + 1}")
        src = np.concatenate(all_src, axis=0)
        dst = np.concatenate(all_dst, axis=0)
        step = rigid_transform_svd(src, dst)
        delta = step @ delta
        rmse = float(np.sqrt(np.mean(np.sum((transform_points(step, src) - dst) ** 2, axis=1))))
        pairs = int(len(src))
        history.append({"iteration": iteration + 1, "rmse_m": rmse, "pairs": pairs, "frames": frame_stats})
        if prev_rmse is not None and abs(prev_rmse - rmse) < 1.0e-5:
            break
        prev_rmse = rmse
    return delta, history


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    world_t_base = load_transform(args.world_T_base_init) if args.world_T_base_init else default_world_t_base()
    wrist3_t_camera = load_transform(args.wrist3_T_camera) if args.wrist3_T_camera else default_wrist3_t_camera_optical()
    frames = load_frame_clouds(args, world_t_base, wrist3_t_camera, rng)
    delta, history = shared_delta_icp(frames, args, rng)
    refined = delta @ world_t_base

    np.save(args.out_dir / "delta_world_T_world_init.npy", delta)
    np.save(args.out_dir / "world_T_base_init.npy", world_t_base)
    np.save(args.out_dir / "world_T_base_refined.npy", refined)

    result = {
        "real_run_dir": str(args.real_run_dir),
        "sim_target_dir": str(args.sim_target_dir),
        "frame_count": len(frames),
        "delta_world_T_world_init": delta.tolist(),
        "delta_translation_m": delta[:3, 3].tolist(),
        "delta_quat_xyzw": matrix_to_quat_xyzw(delta[:3, :3]).tolist(),
        "world_T_base_init": world_t_base.tolist(),
        "world_T_base_refined": refined.tolist(),
        "world_T_base_refined_translation_m": refined[:3, 3].tolist(),
        "world_T_base_refined_quat_xyzw": matrix_to_quat_xyzw(refined[:3, :3]).tolist(),
        "history": history,
    }
    (args.out_dir / "icp_per_frame_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.write_clouds:
        src_initial = np.concatenate([f["source"] for f in frames], axis=0)
        src_refined = transform_points(delta, src_initial)
        target = np.concatenate([f["target"] for f in frames], axis=0)
        write_ply(args.out_dir / "source_initial_world.ply", src_initial, (255, 80, 60))
        write_ply(args.out_dir / "source_refined_world.ply", src_refined, (70, 180, 255))
        write_ply(args.out_dir / "target_world_per_frame_concat.ply", target, (180, 180, 180))

    last = history[-1]
    print(f"frames={len(frames)} pairs={last['pairs']} final_rmse_m={last['rmse_m']:.6f}")
    print(f"delta_translation_m={result['delta_translation_m']}")
    print(f"delta_quat_xyzw={result['delta_quat_xyzw']}")
    print(f"world_T_base_refined_translation_m={result['world_T_base_refined_translation_m']}")
    print(f"wrote {args.out_dir / 'icp_per_frame_result.json'}")


if __name__ == "__main__":
    main()
