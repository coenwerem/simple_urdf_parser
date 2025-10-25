#!/usr/bin/env python3

from simple_urdf_parser.parser import Robot
import numpy as np
import spatialmath as sm
np.random.seed(42) # reproducibility

def main():
    # Get URDF from file and construct a Robot object
    xml_file_path_rel = '../assets/urdf/ur3.urdf'
    robot = Robot(desc_fp=xml_file_path_rel)

    # Get link and joint names of the chain
    link_names = [link._name for link in robot.links]
    joint_names = [joint.name for joint in robot.joints]

    # Get joint origins
    joint_origins_rpyxyz = []
    for joint in robot.joints:
        T: sm.SE3 = joint.origin.T
        rpy = T.rpy(order='xyz', unit='rad')  
        xyz = T.t                             
        joint_origins_rpyxyz.append([rpy, xyz])

    # call the helper printing method
    robot.pretty_print_robot_md(link_names, joint_names, joint_origins_rpyxyz)

    # Get base and EE links
    base_link = robot.base_link
    ee_link = robot.ee_link

    # Constructor for robot configuration
    q_test = robot.Configuration(
                                joints=robot.actuated_joints,
                                joint_values=[0, -np.pi/4, np.pi/4, 0, np.pi/3, 0]
                            )

    # Forward kinematics
    T_fk = robot._compute_fk(
        config=q_test,
        start=base_link._name,
        end=ee_link._name,
        pretty_print=True, 
        round_dp=2
    )
    print("\n")
    print(f"\033[96mT_fk\033[0m:\n{T_fk}\n{'-'*44}")

    # Inverse kinematics
    ik_sol = robot._compute_ik(
        x_d = T_fk,
        init_guess = robot.Configuration.zeros_for_joints(robot.actuated_joints),
        method = 'dls',
        max_iters = 200
    )

    # Verify IK solution
    T_fk_sol = robot._compute_fk(
        config = robot.Configuration(
            robot.actuated_joints,
            joint_values=np.array(ik_sol.joint_values)
        )
    )
   
    T_fk_arr = T_fk.A if isinstance(T_fk, sm.SE3) else np.array(T_fk)
    T_fk_sol_arr = T_fk_sol.A if isinstance(T_fk_sol, sm.SE3) else np.array(T_fk_sol)
    assert np.allclose(T_fk_arr, T_fk_sol_arr, atol=1e-4), "f(ik_sol) must yield original T_fk"
    print(f"\033[96mIK sol\033[0m: {np.round(np.array(ik_sol.joint_values), 3)}\n{'-'*44}")
    print(f"\033[96mT_fk (f(q))\033[0m:\n{T_fk}\n{'-'*44}")
    print(f"\033[96mT_fk_sol (f(ik(f(q))))\033[0m:\n{T_fk_sol}\n{'-'*44}")

    # Check if a configuration is singular with jacobian
    # we use extra guards based on the condition number
    # rather than the Jacobian inverse
    print(f"\033[96mSingularity Tests\033[0m:")
    print(f"{'-'*44}")
    print(q_test.is_singular(robot))
    print(ik_sol.is_singular(robot))

    print(f"{'-'*44}")
    # Jacobian 
    jac = robot._compute_jacobian(
        config=q_test
    )
    print(f"\033[96mJacobian at q={np.round(q_test.joint_values, 4)}\033[0m:\n{np.round(jac, 4)}")

    # Get a random and joint-limit-respecting C-space configuration of the chain
    print(f"{'-'*44}")
    print(f"\033[96mRandom Configuration\033[0m:")
    qrand = robot.Configuration.random_config(robot.actuated_joints, robot)
    print(f"{np.round(qrand.joint_values, 4)}")
    print(f"{'-'*44}")
    

if __name__=='__main__':
    main()