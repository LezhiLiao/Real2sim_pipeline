#!/usr/bin/env python3
"""Refine the robot base pose in an ArUco/world frame with multi-view ICP.

The real capture samples are expected to contain:
  sample_*/pointcloud_camera.npz with points_xyz_m in RealSense optical frame
  sample_*/base_T_tcp.npy with the robot FK pose for the active wrist/TCP frame

The optimization estimates one shared left-multiplied world correction:
  world_T_base_refined = delta_world @ world_T_base_init
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_WORLD_T_BASE_POS = (-0.6144988676, -0.6692399907, 0.0713385308)
DEFAULT_WORLD_T_BASE_QUAT_XYZW = (
    -0.008396469508180033,
    0.037134101814545795,
    0.008116180839915598,
    0.9992420554554332,
)

# USD wrist_3_link/realsense values recorded in CUBE_WRIST_CAMERA_EXTRINSICS.md.
DEFAULT_WRIST3_T_USD_CAMERA_POS = (-0.01836250805, -0.0800952162, 0.07073644041)
DEFAULT_WRIST3_T_USD_CAMERA_QUAT_WXYZ = (
    0.224865558168,
    0.973790609429,
    -0.00306709853,
    -0.034028262449,
)

# RealSense/OpenCV optical frame (+X right, +Y down, +Z forward) to USD camera
# local frame (+X right, +Y up, -Z forward).
USD_CAMERA_R_OPTICAL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a shared ICP correction for the robot world pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-run-dir", type=Path, default=Path("data/icp_run_20260814_155745"))
    parser.add_argument("--target-cloud", type=Path, required=True, help="Simulation/3DGS scene point cloud in world frame (.ply or .npz).")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/icp_refine_robot_world_pose"))
    parser.add_argument("--world-T-base-init", type=Path, default=None, help="Optional 4x4 .npy/.json initial world_T_base.")
    parser.add_argument("--wrist3-T-camera", type=Path, default=None, help="Optional 4x4 .npy/.json wrist3_T_camera_optical.")
    parser.add_argument("--wrist3-camera-is-usd", action="store_true", help="Treat --wrist3-T-camera as wrist3_T_USD_camera and apply USD->optical conversion.")
    parser.add_argument("--sample-glob", default="sample_*")
    parser.add_argument("--max-points-per-sample", type=int, default=30000)
    parser.add_argument("--source-voxel", type=float, default=0.006)
    parser.add_argument("--target-voxel", type=float, default=0.006)
    parser.add_argument("--max-source-points", type=int, default=8000)
    parser.add_argument("--max-target-points", type=int, default=25000)
    parser.add_argument("--max-correspondence-distance", type=float, default=0.04)
    parser.add_argument("--trim-fraction", type=float, default=0.70, help="Keep this closest fraction of valid correspondences each ICP step.")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=1.0)
    parser.add_argument("--exclude-world-box", type=float, nargs=6, action="append", default=[], metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"), help="Remove points inside a world AABB. Repeatable, useful for cube removal.")
    parser.add_argument("--write-clouds", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def quat_xyzw_to_matrix(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return quat_xyzw_to_matrix((x, y, z, w))


def matrix_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    m = np.asarray(r, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    idx = int(np.argmax(np.diag(m)))
    if idx == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    elif idx == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    return q / np.linalg.norm(q)


def make_transform(r: np.ndarray, t: Iterable[float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(r, dtype=np.float64)
    out[:3, 3] = np.asarray(t, dtype=np.float64)
    return out


def transform_points(t: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ t[:3, :3].T + t[:3, 3]


def load_transform(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.shape != (4, 4):
            raise ValueError(f"{path} must contain a 4x4 matrix, got {arr.shape}")
        return arr.astype(np.float64)
    data = json.loads(path.read_text())
    if isinstance(data, list):
        arr = np.asarray(data, dtype=np.float64)
    elif "matrix" in data:
        arr = np.asarray(data["matrix"], dtype=np.float64)
    elif "translation" in data and "quat_xyzw" in data:
        arr = make_transform(quat_xyzw_to_matrix(data["quat_xyzw"]), data["translation"])
    elif "translation" in data and "quat_wxyz" in data:
        arr = make_transform(quat_wxyz_to_matrix(data["quat_wxyz"]), data["translation"])
    else:
        raise ValueError(f"Unsupported transform JSON format: {path}")
    if arr.shape != (4, 4):
        raise ValueError(f"{path} must describe a 4x4 matrix, got {arr.shape}")
    return arr


def default_world_t_base() -> np.ndarray:
    return make_transform(quat_xyzw_to_matrix(DEFAULT_WORLD_T_BASE_QUAT_XYZW), DEFAULT_WORLD_T_BASE_POS)


def default_wrist3_t_camera_optical() -> np.ndarray:
    wrist_t_usd = make_transform(
        quat_wxyz_to_matrix(DEFAULT_WRIST3_T_USD_CAMERA_QUAT_WXYZ),
        DEFAULT_WRIST3_T_USD_CAMERA_POS,
    )
    usd_t_optical = make_transform(USD_CAMERA_R_OPTICAL, (0.0, 0.0, 0.0))
    return wrist_t_usd @ usd_t_optical


def read_npz_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    if "points_xyz_m" in data:
        pts = np.asarray(data["points_xyz_m"], dtype=np.float64)
    elif "points" in data:
        pts = np.asarray(data["points"], dtype=np.float64)
    elif "xyz" in data:
        pts = np.asarray(data["xyz"], dtype=np.float64)
    else:
        raise ValueError(f"{path} has no known point array key")
    colors = np.asarray(data["colors_rgb_uint8"], dtype=np.uint8) if "colors_rgb_uint8" in data else None
    return pts, colors


def read_ply_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1] == "end_header":
                break
        if header_lines[0] != "ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = next((line for line in header_lines if line.startswith("format ")), "")
        vertex_count = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[:1] == ["element"] and parts[1:2] != ["vertex"]:
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"]:
                props.append((parts[1], parts[2]))
        names = [name for _typ, name in props]
        if not {"x", "y", "z"}.issubset(names):
            raise ValueError(f"{path} PLY must contain x/y/z vertex properties")
        if "ascii" in fmt:
            raw = np.loadtxt(f, max_rows=vertex_count)
            pts = raw[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float64)
            colors = None
            if {"red", "green", "blue"}.issubset(names):
                colors = raw[:, [names.index("red"), names.index("green"), names.index("blue")]].astype(np.uint8)
            return pts, colors
        if "binary_little_endian" not in fmt:
            raise ValueError(f"Unsupported PLY format in {path}: {fmt}")
        fmt_map = {"float": "f", "float32": "f", "double": "d", "uchar": "B", "uint8": "B", "int": "i", "uint": "I"}
        struct_fmt = "<" + "".join(fmt_map[typ] for typ, _name in props)
        row_size = struct.calcsize(struct_fmt)
        rows = f.read(row_size * vertex_count)
        arr = np.empty((vertex_count, len(props)), dtype=np.float64)
        for i in range(vertex_count):
            arr[i] = struct.unpack_from(struct_fmt, rows, i * row_size)
        pts = arr[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float64)
        colors = None
        if {"red", "green", "blue"}.issubset(names):
            colors = arr[:, [names.index("red"), names.index("green"), names.index("blue")]].astype(np.uint8)
        return pts, colors


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix == ".npz":
        return read_npz_cloud(path)
    if path.suffix == ".ply":
        return read_ply_cloud(path)
    raise ValueError(f"Unsupported cloud format: {path}")


def clean_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def random_limit(points: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit <= 0 or len(points) <= limit:
        return points
    idx = rng.choice(len(points), size=limit, replace=False)
    return points[idx]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0.0 or len(points) == 0:
        return points
    keys = np.floor(points / float(voxel)).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def remove_world_boxes(points_world: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    if not boxes:
        return points_world
    keep = np.ones(len(points_world), dtype=bool)
    for box in boxes:
        b = np.asarray(box, dtype=np.float64)
        lo = np.minimum(b[:3], b[3:])
        hi = np.maximum(b[:3], b[3:])
        inside = np.all((points_world >= lo) & (points_world <= hi), axis=1)
        keep &= ~inside
    return points_world[keep]


def nearest_neighbors_bruteforce(src: np.ndarray, dst: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    nn_idx = np.empty(len(src), dtype=np.int64)
    nn_dist2 = np.empty(len(src), dtype=np.float64)
    dst_norm = np.einsum("ij,ij->i", dst, dst)
    for start in range(0, len(src), chunk):
        part = src[start : start + chunk]
        dist2 = np.einsum("ij,ij->i", part, part)[:, None] + dst_norm[None, :] - 2.0 * (part @ dst.T)
        idx = np.argmin(dist2, axis=1)
        nn_idx[start : start + len(part)] = idx
        nn_dist2[start : start + len(part)] = dist2[np.arange(len(part)), idx]
    return nn_idx, np.maximum(nn_dist2, 0.0)


def rigid_transform_svd(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    h = src_centered.T @ dst_centered
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = dst_centroid - r @ src_centroid
    return make_transform(r, t)


def icp_point_to_point(
    source_world: np.ndarray,
    target_world: np.ndarray,
    iterations: int,
    max_dist: float,
    trim_fraction: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    transform = np.eye(4, dtype=np.float64)
    history: list[dict[str, float]] = []
    trim_fraction = min(max(float(trim_fraction), 0.05), 1.0)
    max_dist2 = float(max_dist) ** 2
    prev_rmse = None
    for it in range(iterations):
        moved = transform_points(transform, source_world)
        idx, dist2 = nearest_neighbors_bruteforce(moved, target_world)
        valid = dist2 <= max_dist2
        if int(valid.sum()) < 6:
            raise RuntimeError(f"ICP has too few correspondences at iteration {it}: {int(valid.sum())}")
        valid_idx = np.flatnonzero(valid)
        order = valid_idx[np.argsort(dist2[valid])]
        keep_n = max(6, int(math.ceil(len(order) * trim_fraction)))
        keep = order[:keep_n]
        step = rigid_transform_svd(moved[keep], target_world[idx[keep]])
        transform = step @ transform
        rmse = float(np.sqrt(np.mean(dist2[keep])))
        fitness = float(len(keep) / len(source_world))
        history.append({"iteration": it + 1, "rmse_m": rmse, "fitness": fitness, "pairs": int(len(keep))})
        if prev_rmse is not None and abs(prev_rmse - rmse) < 1.0e-5:
            break
        prev_rmse = rmse
    return transform, history


def load_real_source_base(args: argparse.Namespace, wrist3_t_camera: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sample_dirs = sorted(p for p in args.real_run_dir.glob(args.sample_glob) if p.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No sample dirs matched {args.real_run_dir / args.sample_glob}")
    clouds_base = []
    for sample_dir in sample_dirs:
        cloud_path = sample_dir / "pointcloud_camera.npz"
        fk_path = sample_dir / "base_T_tcp.npy"
        if not cloud_path.exists() or not fk_path.exists():
            continue
        points_cam, _colors = read_npz_cloud(cloud_path)
        points_cam = clean_points(points_cam)
        depth = points_cam[:, 2]
        points_cam = points_cam[(depth >= args.min_depth) & (depth <= args.max_depth)]
        points_cam = random_limit(points_cam, args.max_points_per_sample, rng)
        base_t_tcp = np.load(fk_path).astype(np.float64)
        if base_t_tcp.shape != (4, 4):
            raise ValueError(f"{fk_path} must be 4x4, got {base_t_tcp.shape}")
        base_t_camera = base_t_tcp @ wrist3_t_camera
        clouds_base.append(transform_points(base_t_camera, points_cam))
    if not clouds_base:
        raise FileNotFoundError(f"No usable sample point clouds under {args.real_run_dir}")
    merged = np.concatenate(clouds_base, axis=0)
    merged = voxel_downsample(merged, args.source_voxel)
    merged = random_limit(merged, args.max_source_points, rng)
    return merged


def write_ply(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    with path.open("wb") as f:
        f.write(
            (
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
            ).encode("ascii")
        )
        rgb = bytes([int(color[0]), int(color[1]), int(color[2])])
        for p in points:
            f.write(struct.pack("<fff", float(p[0]), float(p[1]), float(p[2])) + rgb)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    world_t_base_init = load_transform(args.world_T_base_init) if args.world_T_base_init else default_world_t_base()
    if args.wrist3_T_camera:
        wrist3_t_camera = load_transform(args.wrist3_T_camera)
        if args.wrist3_camera_is_usd:
            wrist3_t_camera = wrist3_t_camera @ make_transform(USD_CAMERA_R_OPTICAL, (0.0, 0.0, 0.0))
    else:
        wrist3_t_camera = default_wrist3_t_camera_optical()

    source_base = load_real_source_base(args, wrist3_t_camera, rng)
    source_world_init = transform_points(world_t_base_init, source_base)
    source_world_init = remove_world_boxes(source_world_init, args.exclude_world_box)

    target_world, _target_colors = load_cloud(args.target_cloud)
    target_world = clean_points(target_world)
    target_world = remove_world_boxes(target_world, args.exclude_world_box)
    target_world = voxel_downsample(target_world, args.target_voxel)
    target_world = random_limit(target_world, args.max_target_points, rng)

    delta_world, history = icp_point_to_point(
        source_world_init,
        target_world,
        iterations=args.iterations,
        max_dist=args.max_correspondence_distance,
        trim_fraction=args.trim_fraction,
    )
    world_t_base_refined = delta_world @ world_t_base_init
    source_world_refined = transform_points(delta_world, source_world_init)

    np.save(args.out_dir / "delta_world_T_world_init.npy", delta_world)
    np.save(args.out_dir / "world_T_base_init.npy", world_t_base_init)
    np.save(args.out_dir / "world_T_base_refined.npy", world_t_base_refined)
    np.save(args.out_dir / "wrist3_T_camera_optical.npy", wrist3_t_camera)

    result = {
        "real_run_dir": str(args.real_run_dir),
        "target_cloud": str(args.target_cloud),
        "source_points_used": int(len(source_world_init)),
        "target_points_used": int(len(target_world)),
        "delta_world_T_world_init": delta_world.tolist(),
        "delta_translation_m": delta_world[:3, 3].tolist(),
        "delta_quat_xyzw": matrix_to_quat_xyzw(delta_world[:3, :3]).tolist(),
        "world_T_base_init": world_t_base_init.tolist(),
        "world_T_base_refined": world_t_base_refined.tolist(),
        "world_T_base_refined_translation_m": world_t_base_refined[:3, 3].tolist(),
        "world_T_base_refined_quat_xyzw": matrix_to_quat_xyzw(world_t_base_refined[:3, :3]).tolist(),
        "history": history,
        "notes": [
            "Real point clouds are interpreted as RealSense optical frame: +X right, +Y down, +Z forward.",
            "Default wrist3 extrinsic comes from CUBE_WRIST_CAMERA_EXTRINSICS.md and applies USD-camera to optical conversion.",
            "If your base_T_tcp is not wrist_3_link, pass a wrist3_T_camera that matches the saved FK frame.",
        ],
    }
    (args.out_dir / "icp_result.json").write_text(json.dumps(result, indent=2))

    if args.write_clouds:
        write_ply(args.out_dir / "source_initial_world.ply", source_world_init, (255, 80, 60))
        write_ply(args.out_dir / "source_refined_world.ply", source_world_refined, (70, 180, 255))
        write_ply(args.out_dir / "target_world_downsampled.ply", target_world, (180, 180, 180))

    last = history[-1] if history else {"rmse_m": float("nan"), "fitness": 0.0, "pairs": 0}
    print(f"source_points={len(source_world_init)} target_points={len(target_world)}")
    print(f"final_rmse_m={last['rmse_m']:.6f} fitness={last['fitness']:.4f} pairs={last['pairs']}")
    print(f"delta_translation_m={result['delta_translation_m']}")
    print(f"world_T_base_refined_translation_m={result['world_T_base_refined_translation_m']}")
    print(f"wrote {args.out_dir / 'icp_result.json'}")


if __name__ == "__main__":
    main()
