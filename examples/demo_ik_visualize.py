"""IK visualization demo.

Launch, then open http://localhost:8080 in a browser.

Modes:
  default          Animated IK convergence toward a ghost target pose.
  --interactive    Draggable gizmo — robot tracks your target in real time.
  --target_xyz     Custom target position (e.g. --target_xyz 0.2 0.1 0.3).
"""

import argparse
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simple_urdf_parser import Robot
from simple_urdf_parser.visualizer import RobotVisualizer

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "urdf", "ur3.urdf")

parser = argparse.ArgumentParser(description="IK visualization demo")
parser.add_argument(
    "--interactive", action="store_true",
    help="Draggable gizmo mode — solve IK in real time",
)
parser.add_argument(
    "--target_xyz", nargs=3, type=float, metavar=("X", "Y", "Z"),
    help="Target end-effector position (default: FK of a sample config)",
)
args = parser.parse_args()

robot = Robot(desc_fp=URDF_PATH)
viz = RobotVisualizer(robot, port=8080)
viz.build_scene()

if args.interactive:
    if args.target_xyz:
        import spatialmath as sm
        init_pose = sm.SE3(args.target_xyz[0], args.target_xyz[1], args.target_xyz[2])
    else:
        init_pose = None
    viz.add_joint_sliders()
    viz.interactive_ik(init_pose=init_pose)
    print("Viewer ready at http://localhost:8080")
    print("Drag the gizmo to set IK targets. Press Ctrl+C to exit.")
else:
    q_target = Robot.Configuration(
        joints=robot.actuated_joints,
        joint_values=[0.5, -np.pi / 4, np.pi / 3, -0.8, np.pi / 4, 0.2],
    )
    if args.target_xyz:
        import spatialmath as sm
        target_pose = sm.SE3(args.target_xyz[0], args.target_xyz[1], args.target_xyz[2])
    else:
        target_pose = robot._compute_fk(
            config=q_target,
            start=robot.base_link._name,
            end=robot.ee_link._name,
            pretty_print=False,
        )

    viz.show_ghost(q_target, label="target")
    print("Viewer ready at http://localhost:8080")
    print("Starting IK convergence in 3 seconds...")
    time.sleep(3)

    solution = viz.animate_ik(
        target_pose=target_pose,
        method="dls",
        max_iters=300,
        dt=0.02,
        trail=True,
    )
    print(f"IK converged: q = {np.round(np.array(solution.joint_values), 3)}")

print("Press Ctrl+C to exit.")
viz.server.sleep_forever()
