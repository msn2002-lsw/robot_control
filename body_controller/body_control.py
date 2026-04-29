import math
import numpy as np
from typing import List

# 假设这些类在对应模块中已定义好
from robot_base.base_dh import QuadrupedBase
from robot_base.leg_dh import QuadrupedLeg
from robot_base.datatypes import Pose
from kinematics.kinematic import Kinematics
from robot_base.mat_tool import translate_mat, rotate_x_mat, rotate_y_mat, rotate_z_mat

class BodyController:
    def __init__(self, quadruped_base: QuadrupedBase):
        self.base_: QuadrupedBase = quadruped_base

    def pose_command_all(self, foot_positions: List[np.ndarray], req_pose: Pose) -> None:
        """
        对四条腿应用姿态控制。
        :param foot_positions: 包含4个 4x4 numpy 矩阵的列表 (按 LF, RF, LH, RH 顺序)
        :param req_pose: 期望的身体位姿 (Pose)
        """
        for i in range(4):
            # 将每条腿计算后的新位姿矩阵存回列表中
            foot_positions[i] = self.pose_command_single(self.base_.legs[i], req_pose)

    @staticmethod
    def pose_command_single(leg: QuadrupedLeg, req_pose: Pose) -> np.ndarray:
        """
        对单条腿进行姿态逆解。
        返回一个新的 4x4 位姿矩阵。
        """
        # 相对于身体，脚的移动方向与身体的移动方向相反
        req_translation_x = -req_pose.position.x
        req_translation_y = -req_pose.position.y

        # zero_stance 的 Z 往往是个负值（因为脚在身体下方）
        zero_stance_mat = leg.zero_stance()
        zero_z = zero_stance_mat[2, 3]

        # 计算 Z 轴的相对偏移，注意符号反转
        req_translation_z = -(zero_z + req_pose.position.z)
        max_translation_z = -zero_z * 0.65

        # 高度限幅：防止过度伸展或过度下蹲
        if req_translation_z < 0.0:
            req_translation_z = 0.0
        elif req_translation_z > max_translation_z:
            req_translation_z = max_translation_z

        # 从零位站立姿态（4x4 矩阵）开始
        foot_position = np.copy(zero_stance_mat)

        # 1. 施加平移 (相当于 C++ 中的 foot_position.Translate(...))
        t_mat = translate_mat(req_translation_x, req_translation_y, req_translation_z)
        foot_position = foot_position @ t_mat

        # 2. 施加相反的欧拉角旋转 (ZYX 顺规)
        # 旋转身体等价于让脚掌做相反的旋转
        foot_position = foot_position @ rotate_z_mat(-req_pose.orientation.yaw)
        foot_position = foot_position @ rotate_y_mat(-req_pose.orientation.pitch)
        foot_position = foot_position @ rotate_x_mat(-req_pose.orientation.roll)

        # 3. 将全局坐标转换为相对于该腿髋关节的局部坐标
        Kinematics.transform_to_hip(foot_position, leg)

        return foot_position