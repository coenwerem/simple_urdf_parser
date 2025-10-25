#!/usr/bin/env python3

from simple_urdf_parser.parser import Robot
import tempfile, os


def test_parser_loads_urdf():

    with tempfile.NamedTemporaryFile(mode='w+t', delete=True, suffix=".urdf", prefix="my_temp_") as named_temp_file:
        urdf = """<?xml version="1.0" ?>
                    <robot name="test_robot">
                        <link name="base_link"/>
                    </robot>"""
        named_temp_file.write(urdf)
        named_temp_file.flush() # ensure data is flushed to disk before another process re-opens the file.
        print(named_temp_file.name)

        robot = Robot(desc_fp=str(os.path.abspath(named_temp_file.name)))
        assert robot._name == "test_robot"

if __name__=="__main__":
    test_parser_loads_urdf()