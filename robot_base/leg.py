import math
import numpy as np
from typing import List, Optional

# 假设这些是你项目中已经定义好的类
from robot_base.datatypes import GaitConfig
from robot_base.joint import Joint
from robot_base.mat_tool import translate_mat, rotate_x_mat, rotate_y_mat, rotate_z_mat, rpy_to_mat

# ==========================================
# QuadrupedLeg 类定义
# ==========================================
class QuadrupedLeg:
    def __init__(self):
        self._no_of_links: int = 0
        
        # 使用 4x4 单位矩阵初始化 zero_stance
        self._zero_stance: np.ndarray = np.eye(4) 
        self._center_to_nominal: float = 0.0
        
        self._id: int = 0
        self._spine:float = 0
        self._last_touchdown: int = 0
        self._in_contact: bool = True
        
        self._knee_direction: int = 0
        self._is_pantograph: bool = False
        self._gait_phase: bool = True
        
        # 修改关节数量
        self.hip_1: Joint = Joint()
        self.hip_2: Joint = Joint()
        self.upper_leg: Joint = Joint()
        self.lower_leg: Joint = Joint()
        self.ankle_1: Joint = Joint()
        self.ankle_2: Joint = Joint()
        self.foot: Joint = Joint()
        
        self.gait_config: Optional[GaitConfig] = None
        
        self.joint_chain: List[Joint] = [
            self.hip_1,
            self.hip_2,
            self.upper,
            self.lower,
            self.angle_1,
            self.angle_2,
            self.foot,
        ]

    def foot_from_hip(self) -> np.ndarray:
        """正向运动学：计算足端相对于髋关节的 4x4 变换矩阵"""
        # 初始坐标系：髋关节零点
        T_global = np.eye(4)

        # 正向遍历整条运动学链 (假设 index 0 是与髋关节相连的第一个自由度)
        for joint in self.joint_chain:
            
            # 1. 构建静态平移矩阵
            T_trans = translate_mat(joint.x, joint.y, joint.z)
            
            # 2. 构建静态旋转矩阵 (Roll, Pitch, Yaw)
            T_static_rot = rpy_to_mat(joint.roll, joint.pitch, joint.yaw)
            
            # 3. 构建动态关节角旋转矩阵
            T_dynamic_rot = np.eye(4) # 默认无动态旋转 (适用于刚性固定件)
            
            # 使用 elif 防止多次覆盖，同时确保 T_dynamic_rot 永远有定义
            if getattr(joint, '_axis', None):
                if joint._axis.x:
                    T_dynamic_rot = rotate_x_mat(joint.theta)
                elif joint._axis.y:
                    T_dynamic_rot = rotate_y_mat(joint.theta)
                elif joint._axis.z:
                    T_dynamic_rot = rotate_z_mat(joint.theta)

            # 4. 当前关节的完整局部变换矩阵：平移 -> 静态旋转 -> 动态旋转
            T_local = T_trans @ T_static_rot @ T_dynamic_rot
            
            # 5. 全局坐标系右乘局部坐标系
            T_global = T_global @ T_local

        return T_global

    def foot_from_base(self) -> np.ndarray:
        """
        计算足端相对于基座(Base)的完整 4x4 变换矩阵
        计算链：机身中心 -> 髋关节安装位置 -> 髋关节动态旋转 -> 足端
        """
        # 1. 静态平移：机身中心到髋关节安装位置 (安装在机身的哪个角)
        t_static = translate_mat(self.hip.x, self.hip.y, self.hip.z)
        
        # 2. 静态姿态：髋关节在机身上的安装角度 (比如为了展态爬行，可能预设了 90度 Roll)
        # 使用你之前定义的 rpy_to_mat
        r_static = rpy_to_mat(self.hip.roll, self.hip.pitch, self.hip.yaw)
        
        # 3. 组合基座到髋关节中心的静态变换
        T_base_to_hip_origin = t_static @ r_static
        
        # 4. 获取已经计算好的从髋关节旋转中心到足端的变换
        # 注意：确保 foot_from_hip() 内部已经包含了 theta 的旋转
        T_hip_to_foot = self.foot_from_hip()
        
        # 5. 最终连乘：Base -> Hip -> Foot
        # 这种顺序保证了足端坐标能正确反映出相对于机身中心的真实空间位姿
        T_base_to_foot = T_base_to_hip_origin @ T_hip_to_foot
        
        return T_base_to_foot

    def zero_stance(self) -> np.ndarray:
        """计算零位站立姿态，返回一个纯平移的 4x4 矩阵"""
        if self.gait_config is None:
            raise ValueError("GaitConfig is not set")
            
        x = self.hip_1.x + self.ankle_1.x
        y = self.hip_1.y + self.hip_2.y + self.upper_leg.y + self.lower_leg.y
        z = self.hip_1.z + self.upper_leg.z + self.upper_leg.z + self.ankle_1.z + self.ankle_2 + self.foot.z
        
        # 将结果存为 4x4 平移矩阵
        self._zero_stance = translate_mat(x, y, z)
        return self._zero_stance

    def get_center_to_nominal(self) -> float:
        x = self.hip_1.x 
        y = self.hip_1.y
        return math.sqrt(x**2 + y**2)

    # ------------------------------------------
    # 关节设置及 Getter/Setter 保持不变
    # ------------------------------------------
    def set_joints(self, hip_joint_1: float, hip_joint_2: float, upper_leg_joint: float, lower_leg_joint: float, ankle_joint_1: float, ankle_joint_2: float) -> None:
        self.hip_1.theta = hip_joint_1
        self.hip_2.theta = hip_joint_2
        self.upper_leg.theta = upper_leg_joint
        self.lower_leg.theta = lower_leg_joint
        self.angle_1.theta = ankle_joint_1
        self.angle_2.theta = ankle_joint_2

    def set_joints_array(self, joints_array: List[float]) -> None:
        for i in range(7):
            self.joint_chain[i].theta = joints_array[i]

    @property
    def leg_id(self) -> int:
        return self._id

    @leg_id.setter
    def leg_id(self, val: int) -> None:
        self._id = val

    @property
    def last_touchdown(self) -> int:
        return self._last_touchdown

    @last_touchdown.setter
    def last_touchdown(self, current_time: int) -> None:
        self._last_touchdown = current_time

    @property
    def in_contact(self) -> bool:
        return self._in_contact

    @in_contact.setter
    def in_contact(self, val: bool) -> None:
        self._in_contact = val

    @property
    def gait_phase(self) -> bool:
        return self._gait_phase

    @gait_phase.setter
    def gait_phase(self, phase: bool) -> None:
        self._gait_phase = phase

    @property
    def knee_direction(self) -> int:
        return self._knee_direction

    @knee_direction.setter
    def knee_direction(self, direction: int) -> None:
        self._knee_direction = direction

    @property
    def is_pantograph(self) -> bool:
        return self._is_pantograph

    @is_pantograph.setter
    def is_pantograph(self, config: bool) -> None:
        self._is_pantograph = config