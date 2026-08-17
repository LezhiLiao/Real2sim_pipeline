#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

COLLECT_SCRIPT = Path('/data/user/isaacsim6/outputs/baselines/broken_precreated_peg_attempt_20260709/scense_collect_cube_pick_place_embody_tag.py')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, default=Path('/tmp/isaacsim6_outputs/initial_tcp_pose_wristcam_20260816'))
    p.add_argument('--position-base', type=float, nargs=3, default=(-0.01001, -0.48000, 0.35999))
    p.add_argument('--quat-base-wxyz', type=float, nargs=4, default=(0.0, 1.0, 0.0, 0.0))
    p.add_argument('--gripper-command', type=float, default=1.0)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--renderer', default='RayTracedLighting')
    return p.parse_args()


def load_module(args):
    old = sys.argv[:]
    sys.argv = [
        str(COLLECT_SCRIPT), '--mode', 'render', '--output-dir', str(args.output_dir),
        '--width', str(args.width), '--height', str(args.height), '--renderer', args.renderer,
        '--stage-name', 'initial_tcp_pose_scene.usda', '--gripper-close-command', str(args.gripper_command),
    ]
    try:
        spec = importlib.util.spec_from_file_location('cube_collect_scene', COLLECT_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules['cube_collect_scene'] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = args.output_dir / 'debug.txt'
    if debug_path.exists():
        debug_path.unlink()
    module = load_module(args)
    try:
        ctx = module.setup_collect_context(str(args.output_dir), str(debug_path))
        target_pos = np.asarray(args.position_base, dtype=np.float64)
        target_quat = np.asarray(args.quat_base_wxyz, dtype=np.float64)
        joints = module.solve_wrist3_ik_joint_state(
            ctx,
            target_pos,
            target_quat,
            max_iters=120,
            pos_tolerance=0.0005,
            orientation_tolerance_rad=0.003,
        )
        # Force requested closed gripper state after IK.
        dof_positions = ctx.robot.get_dof_positions().numpy().astype(np.float64)
        dof_velocities = ctx.robot.get_dof_velocities().numpy().astype(np.float64)
        if ctx.gripper_dof_indices:
            dof_positions[:, ctx.gripper_dof_indices] = float(args.gripper_command)
            dof_velocities[:, ctx.gripper_dof_indices] = 0.0
        ctx.robot.set_dof_positions(dof_positions)
        ctx.robot.set_dof_velocities(dof_velocities)
        ctx.robot.set_dof_position_targets(dof_positions)
        module.wait_updates(8)
        module.sync_wrist3_usd_to_rigid(ctx, label='initial_tcp_pose_capture')

        actual_pos, actual_quat = module.get_wrist3_pose_base(ctx)
        rp = module.rep.create.render_product(ctx.wrist_camera_path, (args.width, args.height))
        rgb_annotator = module.rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        rgb_annotator.attach([rp])
        module.wait_updates(4)
        for _ in range(4):
            module.simulation_app.update()
        rgb = rgb_annotator.get_data()
        if isinstance(rgb, dict) and 'data' in rgb:
            rgb = rgb['data']
        rgb = np.asarray(rgb)
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            rgb = rgb[..., :3]
        png_path = args.output_dir / 'initial_tcp_wrist_rgb.png'
        module.imageio.imwrite(png_path, rgb.astype(np.uint8))
        meta = {
            'target_position_base': target_pos.tolist(),
            'target_quat_base_wxyz': target_quat.tolist(),
            'actual_position_base': actual_pos.tolist(),
            'actual_quat_base_wxyz': actual_quat.tolist(),
            'solved_arm_joints_rad': joints.tolist(),
            'gripper_command': float(args.gripper_command),
            'wrist_camera_path': ctx.wrist_camera_path,
            'png': str(png_path),
        }
        (args.output_dir / 'initial_tcp_capture_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('png', png_path)
        print('target_position_base', target_pos.tolist())
        print('actual_position_base', actual_pos.tolist())
        print('target_quat_base_wxyz', target_quat.tolist())
        print('actual_quat_base_wxyz', actual_quat.tolist())
        print('solved_arm_joints_rad', joints.tolist())
        return 0
    finally:
        module.simulation_app.close()


if __name__ == '__main__':
    raise SystemExit(main())
