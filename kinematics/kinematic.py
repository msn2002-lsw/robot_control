import math
import numpy as np
from typing import List, Tuple, Optional

# 假设这些类在对应模块中已定义
from robot_base.base_dh import QuadrupedBase
from robot_base.leg_dh import QuadrupedLeg
from robot_base.mat_tool import translate_mat, inverse_transform 

class Kinematics:
    def __init__(self, quadruped_base: QuadrupedBase):
        self.base_: QuadrupedBase = quadruped_base

    def inverse_all(self, foot_positions: List[np.ndarray]) -> Optional[List[float]]:
        """
        计算所有 4 条腿的逆运动学。
        :param foot_positions: 4个表示足端位姿的 4x4 矩阵列表
        :return: 包含 12 个关节角度的列表。如果目标不可达，返回 None。
        """
        calculated_joints = [0.0] * 12

        for i in range(4):
            # 获取单腿的 IK 解
            ik_result = self.inverse_single(self.base_.legs[i], foot_positions[i])
            
            # 检查是否有解（如果超出工作空间，inverse_single 会返回 None 或包含 NaN）
            if ik_result is None or any(math.isnan(angle) for angle in ik_result):
                return None  # 丢弃整个步态计划
                
            spine, hip1, hip2, upper, lower, ankel = ik_result
            calculated_joints[i * 3] = spine
            calculated_joints[i * 3 + 1] = hip1
            calculated_joints[i * 3 + 2] = hip2
            calculated_joints[i * 3 + 3] = upper
            calculated_joints[i * 3 + 4] = lower
            calculated_joints[i * 3 + 5] = ankel

        return calculated_joints

    @staticmethod
    def inverse_single(leg: QuadrupedLeg, foot_position: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """
        TODO: 逆解公式，及其超出工作空间的异常处理，以及需要一定的限位
        计算单腿逆运动学 (核心几何解算)
        :return: [6个关节角]或 [nan, ...] 如果不可达
        """
        
        PX = foot_position[0, 3]
        PY = foot_position[1, 3]
        PZ = foot_position[2, 3]
        print(f"Target foot position: PX={PX}, PY={PY}, PZ={PZ}")

        l1 = leg.hip_1.d
        l2 = leg.spine.a
        l3 = leg.hip_1.a
        l4 = leg.hip_2.a
        l5 = leg.upper_leg.a
        l6 = leg.lower_leg.a
        l7 = leg.ankel.a
        print(f"Leg parameters: l1={l1}, l2={l2}, l3={l3}, l4={l4}, l5={l5}, l6={l6}, l7={l7}")

        #腿部关节角度解算
        # spine_joint, ankel_joint通常不参与逆解计算，直接读取当前角度值
        spine_joint, ankel_joint = leg.spine.theta, leg.ankel.theta+leg.ankel.offset

        # hip1 hip2
        hip1_joint = math.asin((PZ + l7 * math.sin(ankel_joint)) / l3)
        hip2_joint = -hip1_joint

        # lower_leg_joint
        a = l3 / 2 * math.cos(leg.spine.theta + hip1_joint) + (
            l2 + l4) * math.cos(leg.spine.theta) + l1 * math.sin(leg.spine.theta) + l3 / 2 * math.cos(leg.spine.theta - hip1_joint)
        b = l3 / 2 * math.sin(leg.spine.theta + hip1_joint) + (
            l2 + l4) * math.sin(leg.spine.theta) - l1 * math.cos(leg.spine.theta) + l3 / 2 * math.sin(leg.spine.theta - hip1_joint)
        c = l6 + l7 * math.cos(leg.ankel.theta)
        lower_leg_joint = - math.acos(((math.pow((PX - a), 2) + math.pow(
            (PY - b), 2) - math.pow(l5, 2) - math.pow(c, 2))) / (2 * l5 * c))
        lower_leg_joint = lower_leg_joint + leg.lower_leg.offset

        # upper_leg_joint
        d1 = l5 + c * (math.cos(lower_leg_joint) + math.sin(lower_leg_joint))
        d2 = l5 + c * (math.cos(lower_leg_joint) - math.sin(lower_leg_joint))
        upper_leg_joint = math.asin((PX - a + PY - b) / math.sqrt(math.pow(d1, 2) + math.pow(d2, 2))) - math.atan2(d1, d2) - spine_joint
        upper_leg_joint = upper_leg_joint

        return spine_joint, hip1_joint, hip2_joint, upper_leg_joint, lower_leg_joint, ankel_joint

    # ==========================================
    # 正向运动学 (Forward Kinematics)
    # ==========================================
    @staticmethod
    def forward_from_hip(leg: QuadrupedLeg) -> np.ndarray:
        """从髋关节出发计算足端位置（不含hip自身的旋转）"""
        foot_position = np.eye(4)
        # 按链条末端反推：脚端平移 -> 膝关节旋转 -> 小腿平移 -> 髋关节旋转 -> 大腿平移
        foot_position = leg.foot_from_hip()  # 从腿的运动学链计算足端位置
        return foot_position

    @staticmethod
    def forward_from_spine(leg: QuadrupedLeg) -> np.ndarray:
        """从脊柱中心出发计算足端位置（含hip的旋转）"""
        foot_position = np.eye(4)
        foot_position = leg.foot_from_spine()  # 从腿的运动学链计算足端位置
        return foot_position

    @staticmethod
    def forward_from_base(leg: QuadrupedLeg)-> np.ndarray:
        """从基座出发计算完整的足端位置"""
        foot_position = np.eye(4)
        foot_position = leg.foot_from_base()  # 从腿的运动学链计算足端位置
        return foot_position

    @staticmethod
    def transform_to_hip(foot_position: np.ndarray, leg: QuadrupedLeg) -> None:
        """
        TODO: 函数功能暂未完成
        将base系下的坐标转换成hip系下的坐标
        """
        foot_position = foot_position @ inverse_transform(leg.foot_from_hip())

    @staticmethod
    def transform_to_base(foot_position: np.ndarray, leg: QuadrupedLeg) -> None:
        """
        TODO: 函数功能暂未完成
        将hip系下的坐标转换成base系下的坐标
        """
        foot_position = foot_position @ leg.foot_from_hip()