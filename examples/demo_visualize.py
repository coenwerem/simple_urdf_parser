"""Basic robot visualization with interactive joint sliders.

Launch, then open http://localhost:8080 in a browser.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simple_urdf_parser import Robot
from simple_urdf_parser.visualizer import RobotVisualizer

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "urdf", "ur3.urdf")

robot = Robot(desc_fp=URDF_PATH)
viz = RobotVisualizer(robot, port=8080)

viz.build_scene()
viz.add_joint_sliders()

print("Viewer ready at http://localhost:8080")
viz.server.sleep_forever()
