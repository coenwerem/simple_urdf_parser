"""IK convergence visualization with ghost poses and end-effector trail.

Launch, then open http://localhost:8080 in a browser.
Uses a UR3 robot: picks a random reachable target via FK on a random
config, shows a translucent ghost at the target, then animates IK
convergence from zeros.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simple_urdf_parser import Robot
from simple_urdf_parser.visualizer import RobotVisualizer

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "urdf", "ur3.urdf")

robot = Robot(desc_fp=URDF_PATH)
viz = RobotVisualizer(robot, port=8080)

q_target = Robot.Configuration(
    joints=robot.actuated_joints,
    joint_values=[0.5, -np.pi / 4, np.pi / 3, -0.8, np.pi / 4, 0.2],
)

target_pose = robot._compute_fk(
    config=q_target,
    start=robot.base_link._name,
    end=robot.ee_link._name,
    pretty_print=False,
)

viz.build_scene()
viz.show_ghost(q_target, label="target")

print("Viewer ready at http://localhost:8080")
print("Starting IK convergence in 3 seconds...")

import time
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
