import math
import numpy as np
from typing import List, Tuple, Optional

# 假设这些类在对应模块中已定义
from robot_base.base import QuadrupedBase
from robot_base.leg import QuadrupedLeg

# 为了FK使用的辅助矩阵函数 (如果之前已经定义过可以复用)
def translate_mat(x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0,  x ],
        [0.0, 1.0, 0.0,  y ],
        [0.0, 0.0, 1.0,  z ],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotate_y_mat(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [ c,  0.0,  s,  0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0,  c,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotate_x_mat(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0,  c,   -s,  0.0],
        [0.0,  s,    c,  0.0],
        [0.0, 0.0,  0.0, 1.0]
    ])

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
                
            hip, upper, lower = ik_result
            calculated_joints[i * 3] = hip
            calculated_joints[i * 3 + 1] = upper
            calculated_joints[i * 3 + 2] = lower

        return calculated_joints

    @staticmethod
    def inverse_single(leg: QuadrupedLeg, foot_position: np.ndarray) -> Tuple[float, float, float]:
        """
        计算单腿逆运动学 (核心几何解算)
        :return: (hip_joint, upper_leg_joint, lower_leg_joint) 或 (nan, nan, nan) 如果不可达
        """
        # 计算关节链在 Y 轴上的静态偏移累加
        l0 = 0.0
        for i in range(1, 4):
            l0 += leg.joint_chain[i].y

        # 计算大腿 (l1) 和 小腿 (l2) 的有效物理长度及内部几何偏置角
        l1 = -math.sqrt(leg.lower_leg.x**2 + leg.lower_leg.z**2)
        ik_alpha = math.acos(leg.lower_leg.x / l1) - (math.pi / 2.0) 

        l2 = -math.sqrt(leg.foot.x**2 + leg.foot.z**2)
        ik_beta = math.acos(leg.foot.x / l2) - (math.pi / 2.0) 

        # 提取目标位置坐标
        x = foot_position[0, 3]
        y = foot_position[1, 3]
        z = foot_position[2, 3]

        # 1. 求解髋关节侧摆角 (Hip Roll)
        hip_joint = -(math.atan(y / z) - ((math.pi / 2.0) - math.acos(-l0 / math.sqrt(y**2 + z**2))))

        # 2. 坐标系逆变换：将 3D 点反向旋转并平移到大腿-小腿所在的 2D 平面
        # 这里直接手写点绕 X 轴旋转，比操作 4x4 矩阵快得多
        c = math.cos(-hip_joint)
        s = math.sin(-hip_joint)
        
        temp_y = y * c - z * s
        temp_z = y * s + z * c
        
        x = x - leg.upper_leg.x
        y = temp_y
        z = temp_z - leg.upper_leg.z

        # 3. 可达性检查 (Reachability check)
        target_to_foot = math.sqrt(x**2 + z**2)
        if target_to_foot >= (abs(l1) + abs(l2)):
            return float('nan'), float('nan'), float('nan')

        # 4. 余弦定理求解膝关节角 (Knee Pitch)
        lower_leg_joint = leg.knee_direction * math.acos((z**2 + x**2 - l1**2 - l2**2) / (2 * l1 * l2))
        
        # 5. 求解大腿俯仰角 (Thigh Pitch)
        upper_leg_joint = (math.atan(x / z) - math.atan((l2 * math.sin(lower_leg_joint)) / (l1 + (l2 * math.cos(lower_leg_joint)))))
        
        # 补偿几何内部偏置
        lower_leg_joint += ik_beta - ik_alpha
        upper_leg_joint += ik_alpha

        # 6. 异常角度修正，防止大腿向后翻折
        if leg.knee_direction < 0:
            if upper_leg_joint < 0:
                upper_leg_joint += math.pi
        else:
            if upper_leg_joint > 0:
                upper_leg_joint += math.pi

        return hip_joint, upper_leg_joint, lower_leg_joint

    # ==========================================
    # 正向运动学 (Forward Kinematics)
    # ==========================================
    @staticmethod
    def forward_from_hip(leg: QuadrupedLeg, upper_leg_theta: float, lower_leg_theta: float) -> np.ndarray:
        """从髋关节出发计算足端位置（不含hip自身的旋转）"""
        foot_position = np.eye(4)
        
        # 按链条末端反推：脚端平移 -> 膝关节旋转 -> 小腿平移 -> 髋关节旋转 -> 大腿平移
        foot_position = foot_position @ translate_mat(leg.foot.x, leg.foot.y, leg.foot.z)
        foot_position = foot_position @ rotate_y_mat(lower_leg_theta)
        
        foot_position = foot_position @ translate_mat(leg.lower_leg.x, leg.lower_leg.y, leg.lower_leg.z)
        foot_position = foot_position @ rotate_y_mat(upper_leg_theta)
        
        foot_position = foot_position @ translate_mat(leg.upper_leg.x, leg.upper_leg.y, leg.upper_leg.z)
        
        return foot_position

    @staticmethod
    def forward_from_base(leg: QuadrupedLeg, hip_theta: float, upper_leg_theta: float, lower_leg_theta: float) -> np.ndarray:
        """从基座出发计算完整的足端位置"""
        # 复用上一步逻辑
        foot_position = Kinematics.forward_from_hip(leg, upper_leg_theta, lower_leg_theta)
        
        # 增加髋关节的横向旋转和平移
        foot_position = foot_position @ rotate_x_mat(hip_theta)
        foot_position = foot_position @ translate_mat(leg.hip.x, leg.hip.y, leg.hip.z)
        
        return foot_position

    @staticmethod
    def transform_to_hip(foot_position: np.ndarray, leg: QuadrupedLeg) -> None:
        """平移操作：直接修改传入的矩阵 (In-place)"""
        t_mat = translate_mat(-leg.hip.x, -leg.hip.y, -leg.hip.z)
        foot_position[:] = foot_position @ t_mat

    @staticmethod
    def transform_to_base(foot_position: np.ndarray, leg: QuadrupedLeg) -> None:
        """平移操作：直接修改传入的矩阵 (In-place)"""
        t_mat = translate_mat(leg.hip.x, leg.hip.y, leg.hip.z)
        foot_position[:] = foot_position @ t_mat