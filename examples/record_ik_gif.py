"""Record IK convergence GIFs using offscreen pyrender."""

import sys
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
import pyrender
import imageio.v3 as iio
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simple_urdf_parser import Robot

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "urdf", "ur3.urdf")
URDF_DIR = os.path.dirname(os.path.abspath(URDF_PATH))
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

WIDTH, HEIGHT = 800, 600
FPS = 10

TARGETS = {
    "A": [1.0, -1.2, 1.5, -1.0, 0.8, 0.5],
    "B": [-0.7, -0.9, 1.8, -0.4, -0.6, 0.3],
    "C": [0.3, -1.5, 1.0, -1.2, 1.4, -0.8],
}


def load_link_mesh(link, urdf_dir):
    vis = link._visual
    if vis is None or vis.geometry.mesh_dir is None:
        return None, None
    path = os.path.join(urdf_dir, vis.geometry.mesh_dir)
    if not os.path.isfile(path):
        return None, None
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    origin_mat = np.eye(4)
    if vis.origin:
        T = vis.origin.T
        origin_mat[:3, :3] = T.R
        origin_mat[:3, 3] = T.t
    return loaded, origin_mat


def fk_matrix(robot, config, link_name):
    T = robot._compute_fk(
        config=config,
        start=robot.base_link._name,
        end=link_name,
        pretty_print=False,
    )
    mat = np.eye(4)
    mat[:3, :3] = T.R
    mat[:3, 3] = T.t
    return mat


def build_scene(robot, config, link_meshes, ghost_config=None):
    scene = pyrender.Scene(
        bg_color=[25, 25, 30, 255],
        ambient_light=[0.3, 0.3, 0.3],
    )

    for link, (mesh, origin_mat) in link_meshes.items():
        if mesh is None:
            continue
        fk_mat = fk_matrix(robot, config, link)
        pose = fk_mat @ origin_mat
        py_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(py_mesh, pose=pose)

    if ghost_config is not None:
        for link, (mesh, origin_mat) in link_meshes.items():
            if mesh is None:
                continue
            fk_mat = fk_matrix(robot, ghost_config, link)
            pose = fk_mat @ origin_mat
            ghost_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.5, 0.8, 0.5, 0.2],
                alphaMode="BLEND",
            )
            ghost_mesh = pyrender.Mesh.from_trimesh(mesh, material=ghost_mat)
            scene.add(ghost_mesh, pose=pose)

    dir_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [1, -1, 2]
    light_pose[:3, :3] = Rotation.from_euler("xyz", [-45, 30, 0], degrees=True).as_matrix()
    scene.add(dir_light, pose=light_pose)

    fill_light = pyrender.DirectionalLight(color=[0.7, 0.75, 0.9], intensity=1.5)
    fill_pose = np.eye(4)
    fill_pose[:3, 3] = [-1, 1, 1]
    fill_pose[:3, :3] = Rotation.from_euler("xyz", [30, -150, 0], degrees=True).as_matrix()
    scene.add(fill_light, pose=fill_pose)

    ground = trimesh.creation.box(extents=[1.5, 1.5, 0.005])
    ground_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.2, 0.2, 0.22, 1.0],
    )
    ground_mesh = pyrender.Mesh.from_trimesh(ground, material=ground_mat, smooth=False)
    ground_pose = np.eye(4)
    ground_pose[2, 3] = -0.003
    scene.add(ground_mesh, pose=ground_pose)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0.8, -0.7, 0.5]
    look_at = np.array([0.0, 0.0, 0.18])
    fwd = look_at - cam_pose[:3, 3]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cam_pose[:3, :3] = np.column_stack([right, up, -fwd])
    scene.add(cam, pose=cam_pose)

    return scene


def run_ik(robot, link_meshes, renderer, q_target, label):
    target_pose = robot._compute_fk(
        config=q_target,
        start=robot.base_link._name,
        end=robot.ee_link._name,
        pretty_print=False,
    )

    config = Robot.Configuration.zeros_for_joints(robot.actuated_joints)
    eps = 1e-4
    gamma = 0.3
    damp_factor = 0.01
    max_iters = 300
    frames = []
    prev_err_norm = float("inf")
    stall_count = 0

    for i in range(max_iters):
        ee_T = robot._compute_fk(
            config=config,
            start=robot.base_link._name,
            end=robot.ee_link._name,
            pretty_print=False,
        )

        pos_err = target_pose.t - ee_T.t
        rot_err = Robot.compute_tsp_rot_error(target_pose, ee_T)
        err = np.concatenate([pos_err, rot_err])
        err_norm = np.linalg.norm(err)

        scene = build_scene(robot, config, link_meshes, ghost_config=q_target)
        color, _ = renderer.render(scene)
        frames.append(color.copy())

        if i % 5 == 0:
            print(f"  [{label}] iter {i:3d}  err={err_norm:.5f}  frames={len(frames)}")

        if err_norm < eps:
            print(f"  [{label}] Converged at iteration {i}")
            break

        if abs(prev_err_norm - err_norm) < 1e-6:
            stall_count += 1
            if stall_count > 10:
                print(f"  [{label}] Stalled at iteration {i}")
                break
        else:
            stall_count = 0
        prev_err_norm = err_norm

        J = robot._compute_jacobian(config=config)
        dq = J.T @ np.linalg.solve(J @ J.T + damp_factor**2 * np.eye(6), err)
        new_vals = np.array(config.joint_values) + gamma * dq
        config = Robot.Configuration(
            joints=robot.actuated_joints,
            joint_values=new_vals.tolist(),
        )
        config = Robot.clamp_limits(config)

    # Hold final pose
    scene = build_scene(robot, config, link_meshes, ghost_config=q_target)
    color, _ = renderer.render(scene)
    for _ in range(15):
        frames.append(color.copy())

    return frames


print("Loading robot and meshes ...")
robot = Robot(desc_fp=URDF_PATH)

link_meshes = {}
for link in robot.links:
    try:
        robot._get_joint_to_parent(link._name)
    except ValueError:
        pass
    mesh, origin = load_link_mesh(link, URDF_DIR)
    link_meshes[link._name] = (mesh, origin if origin is not None else np.eye(4))

renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)

for label, joint_vals in TARGETS.items():
    print(f"\n--- Target {label} ---")
    q_target = Robot.Configuration(
        joints=robot.actuated_joints,
        joint_values=joint_vals,
    )
    frames = run_ik(robot, link_meshes, renderer, q_target, label)
    out_path = os.path.join(OUT_DIR, f"demo_ik_{label}.gif")
    print(f"  {len(frames)} frames -> {out_path}")
    iio.imwrite(out_path, frames, duration=1000 // FPS, loop=0)

renderer.delete()
print("\nDone.")
