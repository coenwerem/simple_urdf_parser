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


_LINK_COLORS = [
    (70, 130, 180),   # steel blue
    (100, 149, 237),  # cornflower
    (72, 118, 176),   # muted blue
    (80, 140, 200),   # sky
    (90, 120, 190),   # periwinkle
    (65, 145, 165),   # teal-ish
]

_JOINT_COLOR = (220, 60, 60)
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

        for joint in self.robot.actuated_joints:
            T = self.robot._compute_fk(
                config=config,
                start=self.robot.base_link._name,
                end=joint.child._name,
                pretty_print=False,
            )
            wxyz, pos = _se3_to_viser(T)
            self.server.scene.add_icosphere(
                f"/robot/joints/{joint.name}",
                radius=0.015,
                color=_JOINT_COLOR,
                wxyz=wxyz,
                position=pos,
            )

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

            for joint in self.robot.actuated_joints:
                T = self.robot._compute_fk(
                    config=config,
                    start=self.robot.base_link._name,
                    end=joint.child._name,
                    pretty_print=False,
                )
                wxyz, pos = _se3_to_viser(T)
                handle = self.server.scene.get_handle_by_name(
                    f"/robot/joints/{joint.name}"
                )
                if handle is not None:
                    handle.wxyz = wxyz
                    handle.position = pos

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

    def show_ghost(
        self, config: Robot.Configuration, label: str = "ghost"
    ) -> None:
        """Show a translucent ghost of the robot at a given configuration."""
        for link in self.robot.links:
            try:
                self.robot._get_joint_to_parent(link._name)
            except ValueError:
                continue
            T = self.robot._compute_fk(
                config=config,
                start=self.robot.base_link._name,
                end=link._name,
                pretty_print=False,
            )
            wxyz, pos = _se3_to_viser(T)

            vis = link._visual
            if vis is None:
                continue

            geom = vis.geometry
            if geom.geom_type == "cylinder":
                self.server.scene.add_cylinder(
                    f"/ghost_{label}/{link._name}",
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
                    f"/ghost_{label}/{link._name}",
                    color=(150, 150, 150),
                    dimensions=dims,
                    opacity=_GHOST_ALPHA,
                    wxyz=wxyz,
                    position=pos,
                )
            elif geom.geom_type == "sphere":
                self.server.scene.add_icosphere(
                    f"/ghost_{label}/{link._name}",
                    radius=geom.radius,
                    color=(150, 150, 150),
                    opacity=_GHOST_ALPHA,
                    wxyz=wxyz,
                    position=pos,
                )

    def _add_link_geometry(self, link, color: tuple) -> None:
        """Add visual geometry for a link."""
        vis = link._visual
        if vis is None:
            return

        geom = vis.geometry
        mat_color = color
        if vis.material and vis.material.color:
            c = vis.material.color
            mat_color = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

        name = f"/robot/{link._name}/visual"
        if geom.geom_type == "cylinder":
            self.server.scene.add_cylinder(
                name,
                radius=geom.radius,
                height=geom.height,
                color=mat_color,
            )
        elif geom.geom_type == "box":
            dims = tuple(geom.geometry.extents)
            self.server.scene.add_box(name, color=mat_color, dimensions=dims)
        elif geom.geom_type == "sphere":
            self.server.scene.add_icosphere(
                name, radius=geom.radius, color=mat_color
            )
        elif geom.geom_type == "mesh":
            dims = tuple(geom.geometry.extents)
            self.server.scene.add_box(name, color=mat_color, dimensions=dims)

        self._geom_handles[link._name] = name

    @staticmethod
    def _identity_se3():
        import spatialmath as sm

        return sm.SE3()
