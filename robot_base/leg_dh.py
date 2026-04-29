import math
import numpy as np
from typing import List, Optional

# 假设这些是你项目中已经定义好的类
from robot_base.datatypes import GaitConfig
from robot_base.joint_dh import Joint
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
        self._last_touchdown: int = 0
        self._in_contact: bool = True
        self._knee_direction: int = 1  # 1 或 -1，取决于膝关节的弯曲方向
        
        self._gait_phase: bool = True
        
        # 初始化腿部关节
        self.spine: Joint = Joint()  
        self.hip_1: Joint = Joint()
        self.hip_2: Joint = Joint()
        self.upper_leg: Joint = Joint()
        self.lower_leg: Joint = Joint()
        self.ankel: Joint = Joint()

        self.foot: Joint = Joint()
        
        self.gait_config: Optional[GaitConfig] = None
        
        self.joint_chain: List[Joint] = [
            self.spine,
            self.hip_1,
            self.hip_2,
            self.upper_leg,
            self.lower_leg,
            self.ankel,
            self.foot,
        ]
    
    def spine_to_base(self) -> np.ndarray:
        """计算从脊柱中心到基座坐标系的变换矩阵"""
        T = np.eye(4)

        if self._id == 0:  # 左前腿
            T = rotate_z_mat(np.pi/2)
        elif self._id == 1:  # 右前腿
            T = rotate_z_mat(-np.pi/2)
        elif self._id == 2:  # 左后腿
            T = rotate_z_mat(np.pi/2)
        elif self._id == 3:  # 右后腿
            T = rotate_z_mat(-np.pi/2)

        return T

    def foot_from_hip(self) -> np.ndarray:
        """正向运动学：计算足端相对于髋关节的 4x4 变换矩阵, 用于站立式"""
        # 初始坐标系：髋关节零点
        T_global = np.eye(4)

        # 正向遍历整条运动学链 (假设 index 0 是与髋关节相连的第一个自由度)
        for joint in self.joint_chain[1:]:  # 从 hip_2 开始，因为 hip_1 是髋关节的零点
            T_global = T_global @ joint.transform()
        return T_global

    def foot_from_spine(self) -> np.ndarray:
        """
        计算足端相对于基座(脊柱中心)的完整 4x4 变换矩阵
        计算链：脊柱中心 -> 髋关节安装位置 -> 髋关节动态旋转 -> 足端
        """
        # 正向遍历整条运动学链 (假设 index 0 是与髋关节相连的第一个自由度)
        T_spine_to_foot = np.eye(4)  # 从基座到足端的变换矩阵初始为单位矩阵 

        for joint in self.joint_chain:
            T_spine_to_foot = T_spine_to_foot @ joint.transform()

        return T_spine_to_foot

    def foot_from_base(self) -> np.ndarray:
        """
        计算足端相对于基座(脊柱中心)的完整 4x4 变换矩阵
        计算链：基座 -> 脊柱中心 -> 髋关节安装位置 -> 髋关节动态旋转 -> 足端
        """
        T_base_to_foot = np.eye(4)  # 设置初始值 

        # 先将基座标系转换成脊柱坐标系（基座标系方向：X轴-机器人前方，Y轴-机器人左侧，Z轴-机器人上方:w）
        if self._id == 0:  # 左前腿
            T_base_to_foot = T_base_to_foot @ rotate_z_mat(np.pi/2)  
        elif self._id == 1:  # 右前腿
            T_base_to_foot = T_base_to_foot @ rotate_z_mat(-np.pi/2)  
        elif self._id == 2:  # 左后腿
            T_base_to_foot = T_base_to_foot @ rotate_z_mat(np.pi/2)  
        elif self._id == 3:  # 右后腿
            T_base_to_foot = T_base_to_foot @ rotate_z_mat(-np.pi/2)

        # 遍历其余关节
        for joint in self.joint_chain:
            T_base_to_foot = T_base_to_foot @ joint.transform()

        return T_base_to_foot

    def zero_stance(self) -> np.ndarray:
        """
        TODO: 修改为机器人运动初始状态计算, 目前以初始状态代替
        零位站立姿态，返回一个纯平移的 4x4 矩阵
        """

        if self.gait_config is None:
            raise ValueError("GaitConfig is not set")
            
        x = self.hip_2.a + self.upper_leg.a + self.lower_leg.a + self.ankel.a
        y = self.hip_1.d + self.ankel.a
        z = self.ankel.a
        # 将结果存为 4x4 平移矩阵
        self._zero_stance = translate_mat(x, y, z)
        return self.foot_from_base()  # 返回足端相对于基座坐标系的零位站立姿态

    def get_center_to_nominal(self) -> float:
        x = self.spine.a 
        y = self.hip_1.d
        return math.sqrt(x**2 + y**2)

    # ------------------------------------------
    # 关节设置及 Getter/Setter 保持不变
    # ------------------------------------------
    def set_joints(self, spine_joint: float, hip_joint_1: float, hip_joint_2: float, upper_leg_joint: float, lower_leg_joint: float, ankel_joint: float) -> None:
        self.spine.theta = spine_joint
        self.hip_1.theta = hip_joint_1
        self.hip_2.theta = hip_joint_2
        self.upper_leg.theta = upper_leg_joint
        self.lower_leg.theta = lower_leg_joint
        self.ankel.theta = ankel_joint

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
