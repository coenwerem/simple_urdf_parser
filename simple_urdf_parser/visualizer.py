"""Viser-based 3D visualization for simple_urdf_parser robots."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    import viser

from .parser import Robot


def _se3_to_viser(T) -> tuple[tuple, tuple]:
    """Convert an SE3 transform to viser (wxyz, position) tuples."""
    r = Rotation.from_matrix(T.R)
    q = r.as_quat()  # x, y, z, w
    wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    pos = tuple(float(v) for v in T.t)
    return wxyz, pos


def _viser_to_se3(wxyz, position):
    """Convert viser (wxyz, position) back to an SE3 transform."""
    import spatialmath as sm

    q_xyzw = [float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])]
    R = Rotation.from_quat(q_xyzw).as_matrix()
    t = np.array([float(v) for v in position])
    return sm.SE3.Rt(R, t)


_LINK_COLORS = [
    (70, 130, 180),   # steel blue
    (100, 149, 237),  # cornflower
    (72, 118, 176),   # muted blue
    (80, 140, 200),   # sky
    (90, 120, 190),   # periwinkle
    (65, 145, 165),   # teal-ish
]

_EE_COLOR = (50, 205, 50)
_GHOST_ALPHA = 0.15
_TRAIL_COLOR = (255, 200, 50)


class RobotVisualizer:
    """Interactive 3D robot visualizer using viser.

    Parameters
    ----------
    robot : Robot
        A parsed Robot instance.
    host : str
        Viser server host.
    port : int
        Viser server port.
    """

    def __init__(self, robot: Robot, host: str = "0.0.0.0", port: int = 8080):
        import viser

        self.robot = robot
        self.server = viser.ViserServer(host=host, port=port)
        self._frame_handles: dict[str, viser.FrameHandle] = {}
        self._geom_handles: dict[str, object] = {}
        self._joint_sliders: dict[str, viser.GuiSliderHandle] = {}

        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/grid", width=2.0, height=2.0)

    def build_scene(self, config: Robot.Configuration | None = None) -> None:
        """Build the full robot scene from link/joint structure.

        Parameters
        ----------
        config : Robot.Configuration, optional
            Joint configuration to display. Defaults to zeros.
        """
        if config is None:
            config = Robot.Configuration.zeros_for_joints(self.robot.actuated_joints)

        for link in self.robot.links:
            try:
                joint = self.robot._get_joint_to_parent(link._name)
            except ValueError:
                joint = None
            if joint is None:
                T = self._identity_se3()
            else:
                T = self.robot._compute_fk(
                    config=config,
                    start=self.robot.base_link._name,
                    end=link._name,
                    pretty_print=False,
                )

            wxyz, pos = _se3_to_viser(T)
            frame = self.server.scene.add_frame(
                f"/robot/{link._name}",
                show_axes=False,
                wxyz=wxyz,
                position=pos,
            )
            self._frame_handles[link._name] = frame

            color = _LINK_COLORS[
                self.robot.links.index(link) % len(_LINK_COLORS)
            ]
            self._add_link_geometry(link, color)

        ee_T = self.robot._compute_fk(
            config=config,
            start=self.robot.base_link._name,
            end=self.robot.ee_link._name,
            pretty_print=False,
        )
        wxyz, pos = _se3_to_viser(ee_T)
        self.server.scene.add_frame(
            "/robot/ee_frame",
            show_axes=True,
            axes_length=0.08,
            axes_radius=0.004,
            wxyz=wxyz,
            position=pos,
        )

    def update_config(self, config: Robot.Configuration) -> None:
        """Update the scene to reflect a new configuration."""
        with self.server.atomic():
            for link in self.robot.links:
                try:
                    joint = self.robot._get_joint_to_parent(link._name)
                except ValueError:
                    continue
                T = self.robot._compute_fk(
                    config=config,
                    start=self.robot.base_link._name,
                    end=link._name,
                    pretty_print=False,
                )
                wxyz, pos = _se3_to_viser(T)
                if link._name in self._frame_handles:
                    self._frame_handles[link._name].wxyz = wxyz
                    self._frame_handles[link._name].position = pos

            ee_T = self.robot._compute_fk(
                config=config,
                start=self.robot.base_link._name,
                end=self.robot.ee_link._name,
                pretty_print=False,
            )
            wxyz, pos = _se3_to_viser(ee_T)
            ee_handle = self.server.scene.get_handle_by_name("/robot/ee_frame")
            if ee_handle is not None:
                ee_handle.wxyz = wxyz
                ee_handle.position = pos

    def add_joint_sliders(self) -> None:
        """Add GUI sliders for each actuated joint."""
        with self.server.gui.add_folder("Joint Controls"):
            for joint in self.robot.actuated_joints:
                lo, hi = joint.limits
                if lo == hi:
                    lo, hi = -np.pi, np.pi
                slider = self.server.gui.add_slider(
                    label=joint.name,
                    min=float(lo),
                    max=float(hi),
                    step=0.01,
                    initial_value=0.0,
                )
                self._joint_sliders[joint.name] = slider

                @slider.on_update
                def _on_change(event, _s=slider, _j=joint) -> None:
                    vals = [
                        self._joint_sliders[j.name].value
                        for j in self.robot.actuated_joints
                    ]
                    cfg = Robot.Configuration(
                        joints=self.robot.actuated_joints,
                        joint_values=vals,
                    )
                    self.update_config(cfg)

    def animate_configs(
        self,
        configs: Sequence[Robot.Configuration],
        dt: float = 0.05,
        loop: bool = True,
        trail: bool = False,
    ) -> None:
        """Animate through a sequence of configurations.

        Parameters
        ----------
        configs : sequence of Configuration
            Ordered configurations to play.
        dt : float
            Seconds between frames.
        loop : bool
            Whether to loop indefinitely.
        trail : bool
            If True, draw end-effector trail as point cloud.
        """
        trail_points = []
        running = True
        while running:
            for cfg in configs:
                self.update_config(cfg)
                if trail:
                    ee_T = self.robot._compute_fk(
                        config=cfg,
                        start=self.robot.base_link._name,
                        end=self.robot.ee_link._name,
                        pretty_print=False,
                    )
                    trail_points.append(ee_T.t.copy())
                    if len(trail_points) > 1:
                        pts = np.array(trail_points)
                        colors = np.tile(_TRAIL_COLOR, (len(pts), 1)).astype(
                            np.uint8
                        )
                        self.server.scene.add_point_cloud(
                            "/trail",
                            points=pts,
                            colors=colors,
                            point_size=0.005,
                        )
                time.sleep(dt)
            if not loop:
                running = False

    def animate_ik(
        self,
        target_pose,
        init_config: Robot.Configuration | None = None,
        method: str = "jacinv",
        max_iters: int = 200,
        dt: float = 0.03,
        trail: bool = True,
    ) -> Robot.Configuration:
        """Visualize IK convergence step by step.

        Parameters
        ----------
        target_pose : SE3
            Desired end-effector pose.
        init_config : Configuration, optional
            Starting configuration. Defaults to zeros.
        method : str
            IK method ('jacinv' or 'dls').
        max_iters : int
            Maximum iterations.
        dt : float
            Delay between rendered iterations.
        trail : bool
            Draw end-effector trajectory.

        Returns
        -------
        Configuration
            The final IK solution.
        """
        import spatialmath as sm

        if init_config is None:
            init_config = Robot.Configuration.zeros_for_joints(
                self.robot.actuated_joints
            )

        wxyz, pos = _se3_to_viser(target_pose)
        self.server.scene.add_frame(
            "/ik_target",
            show_axes=True,
            axes_length=0.1,
            axes_radius=0.006,
            origin_radius=0.012,
            origin_color=(255, 80, 80),
            wxyz=wxyz,
            position=pos,
        )

        eps = 1e-4
        gamma = 1.0
        damp_factor = 0.01 if method == "dls" else 0.0
        config = init_config.copy()
        trail_points = []

        for i in range(max_iters):
            self.update_config(config)

            ee_T = self.robot._compute_fk(
                config=config,
                start=self.robot.base_link._name,
                end=self.robot.ee_link._name,
                pretty_print=False,
            )

            if trail:
                trail_points.append(ee_T.t.copy())
                if len(trail_points) > 1:
                    pts = np.array(trail_points)
                    colors = np.tile(
                        _TRAIL_COLOR, (len(pts), 1)
                    ).astype(np.uint8)
                    self.server.scene.add_point_cloud(
                        "/ik_trail",
                        points=pts,
                        colors=colors,
                        point_size=0.004,
                    )

            pos_err = target_pose.t - ee_T.t
            rot_err = Robot.compute_tsp_rot_error(target_pose, ee_T)
            err = np.concatenate([pos_err, rot_err])

            if np.linalg.norm(err) < eps:
                break

            J = self.robot._compute_jacobian(config=config)
            if method == "dls":
                dq = J.T @ np.linalg.solve(
                    J @ J.T + damp_factor**2 * np.eye(6), err
                )
            else:
                dq = np.linalg.pinv(J) @ err

            new_vals = np.array(config.joint_values) + gamma * dq
            config = Robot.Configuration(
                joints=self.robot.actuated_joints,
                joint_values=new_vals.tolist(),
            )
            config = Robot.clamp_limits(config)
            time.sleep(dt)

        self.update_config(config)
        return config

    def interactive_ik(
        self,
        init_pose=None,
        init_config: Robot.Configuration | None = None,
        method: str = "dls",
        max_iters: int = 15,
    ) -> None:
        """Add a draggable target gizmo that solves IK in real time.

        Parameters
        ----------
        init_pose : SE3, optional
            Initial gizmo pose. Defaults to current EE pose.
        init_config : Configuration, optional
            Starting joint configuration. Defaults to zeros.
        method : str
            IK method ('jacinv' or 'dls').
        max_iters : int
            Max IK iterations per drag update (low is fine — warm-started).
        """
        if init_config is None:
            init_config = Robot.Configuration.zeros_for_joints(
                self.robot.actuated_joints
            )
        self._ik_config = init_config.copy()

        if init_pose is None:
            init_pose = self.robot._compute_fk(
                config=self._ik_config,
                start=self.robot.base_link._name,
                end=self.robot.ee_link._name,
                pretty_print=False,
            )

        wxyz, pos = _se3_to_viser(init_pose)
        gizmo = self.server.scene.add_transform_controls(
            "/ik_target",
            scale=0.15,
            wxyz=wxyz,
            position=pos,
        )

        eps = 1e-4
        gamma = 1.0
        damp_factor = 0.01 if method == "dls" else 0.0

        @gizmo.on_update
        def _on_drag(event) -> None:
            target = _viser_to_se3(gizmo.wxyz, gizmo.position)
            config = self._ik_config

            for _ in range(max_iters):
                ee_T = self.robot._compute_fk(
                    config=config,
                    start=self.robot.base_link._name,
                    end=self.robot.ee_link._name,
                    pretty_print=False,
                )
                pos_err = target.t - ee_T.t
                rot_err = Robot.compute_tsp_rot_error(target, ee_T)
                err = np.concatenate([pos_err, rot_err])
                if np.linalg.norm(err) < eps:
                    break

                J = self.robot._compute_jacobian(config=config)
                if method == "dls":
                    dq = J.T @ np.linalg.solve(
                        J @ J.T + damp_factor**2 * np.eye(6), err
                    )
                else:
                    dq = np.linalg.pinv(J) @ err

                new_vals = np.array(config.joint_values) + gamma * dq
                config = Robot.Configuration(
                    joints=self.robot.actuated_joints,
                    joint_values=new_vals.tolist(),
                )
                config = Robot.clamp_limits(config)

            self._ik_config = config
            self.update_config(config)

    def show_ghost(
        self, config: Robot.Configuration, label: str = "ghost"
    ) -> None:
        """Show a translucent ghost of the robot at a given configuration."""
        for link in self.robot.links:
            try:
                self.robot._get_joint_to_parent(link._name)
            except ValueError:
                continue
            T_link = self.robot._compute_fk(
                config=config,
                start=self.robot.base_link._name,
                end=link._name,
                pretty_print=False,
            )

            vis = link._visual
            if vis is None:
                continue

            T_vis = T_link * vis.origin.T if vis.origin else T_link
            wxyz, pos = _se3_to_viser(T_vis)

            geom = vis.geometry
            ghost_name = f"/ghost_{label}/{link._name}"

            if geom.geom_type == "mesh" and geom.mesh_dir:
                mesh = self._load_mesh(geom.mesh_dir)
                if mesh is not None:
                    self.server.scene.add_mesh_simple(
                        ghost_name,
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        color=(150, 150, 150),
                        opacity=_GHOST_ALPHA,
                        wxyz=wxyz,
                        position=pos,
                    )
                    continue

            if geom.geom_type == "cylinder":
                self.server.scene.add_cylinder(
                    ghost_name,
                    radius=geom.radius,
                    height=geom.height,
                    color=(150, 150, 150),
                    opacity=_GHOST_ALPHA,
                    wxyz=wxyz,
                    position=pos,
                )
            elif geom.geom_type == "box":
                dims = tuple(geom.geometry.extents)
                self.server.scene.add_box(
                    ghost_name,
                    color=(150, 150, 150),
                    dimensions=dims,
                    opacity=_GHOST_ALPHA,
                    wxyz=wxyz,
                    position=pos,
                )
            elif geom.geom_type == "sphere":
                self.server.scene.add_icosphere(
                    ghost_name,
                    radius=geom.radius,
                    color=(150, 150, 150),
                    opacity=_GHOST_ALPHA,
                    wxyz=wxyz,
                    position=pos,
                )

    def _add_link_geometry(self, link, color: tuple) -> None:
        """Add visual geometry for a link, applying the visual origin offset."""
        vis = link._visual
        if vis is None:
            return

        geom = vis.geometry
        mat_color = color
        if vis.material and vis.material.color:
            c = vis.material.color
            mat_color = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

        vo_wxyz, vo_pos = (1, 0, 0, 0), (0, 0, 0)
        if vis.origin:
            vo_wxyz, vo_pos = _se3_to_viser(vis.origin.T)

        name = f"/robot/{link._name}/visual"

        if geom.geom_type == "mesh" and geom.mesh_dir:
            mesh = self._load_mesh(geom.mesh_dir)
            if mesh is not None:
                self.server.scene.add_mesh_trimesh(
                    name, mesh, wxyz=vo_wxyz, position=vo_pos
                )
                self._geom_handles[link._name] = name
                return

        if geom.geom_type == "cylinder":
            self.server.scene.add_cylinder(
                name,
                radius=geom.radius,
                height=geom.height,
                color=mat_color,
                wxyz=vo_wxyz,
                position=vo_pos,
            )
        elif geom.geom_type == "box":
            dims = tuple(geom.geometry.extents)
            self.server.scene.add_box(
                name, color=mat_color, dimensions=dims,
                wxyz=vo_wxyz, position=vo_pos,
            )
        elif geom.geom_type == "sphere":
            self.server.scene.add_icosphere(
                name, radius=geom.radius, color=mat_color,
                wxyz=vo_wxyz, position=vo_pos,
            )
        else:
            dims = tuple(geom.geometry.extents)
            self.server.scene.add_box(
                name, color=mat_color, dimensions=dims,
                wxyz=vo_wxyz, position=vo_pos,
            )

        self._geom_handles[link._name] = name

    def _load_mesh(self, mesh_path: str):
        """Resolve a URDF mesh path and load it as a trimesh.Trimesh."""
        import os
        import trimesh as _trimesh

        urdf_dir = os.path.dirname(os.path.abspath(self.robot._desc_fp))
        full_path = os.path.join(urdf_dir, mesh_path)

        if not os.path.isfile(full_path):
            return None

        try:
            loaded = _trimesh.load(full_path)
            if isinstance(loaded, _trimesh.Scene):
                return _trimesh.util.concatenate(list(loaded.geometry.values()))
            return loaded
        except Exception:
            return None

    @staticmethod
    def _identity_se3():
        import spatialmath as sm

        return sm.SE3()
