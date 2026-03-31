import math
import numpy as np
from typing import List, Optional

# 假设这些是你项目中已经定义好的类
from .datatypes import GaitConfig
from .joint import Joint

# ==========================================
# 齐次变换矩阵辅助函数 (替代 geometry::Transformation 的底层逻辑)
# ==========================================
def translate_mat(x: float, y: float, z: float) -> np.ndarray:
    """生成 4x4 平移矩阵"""
    return np.array([
        [1.0, 0.0, 0.0,  x ],
        [0.0, 1.0, 0.0,  y ],
        [0.0, 0.0, 1.0,  z ],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotate_x_mat(theta: float) -> np.ndarray:
    """生成绕 X 轴旋转的 4x4 矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0,  c,   -s,  0.0],
        [0.0,  s,    c,  0.0],
        [0.0, 0.0,  0.0, 1.0]
    ])

def rotate_y_mat(theta: float) -> np.ndarray:
    """生成绕 Y 轴旋转的 4x4 矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c,  0.0,  s,  0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0,  c,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

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
        
        self._knee_direction: int = 0
        self._is_pantograph: bool = False
        self._gait_phase: bool = True
        
        self.hip: Joint = Joint()
        self.upper_leg: Joint = Joint()
        self.lower_leg: Joint = Joint()
        self.foot: Joint = Joint()
        
        self.gait_config: Optional[GaitConfig] = None
        
        self.joint_chain: List[Joint] = [
            self.hip,
            self.upper_leg,
            self.lower_leg,
            self.foot
        ]

    def foot_from_hip(self) -> np.ndarray:
        """正向运动学：计算足端相对于髋关节的 4x4 变换矩阵"""
        # 初始化为单位矩阵 Identity<4,4>()
        foot_position = np.eye(4)

        for i in range(3, 0, -1):
            current_joint = self.joint_chain[i]
            prev_joint = self.joint_chain[i - 1]
            
            # foot_position.Translate(...)
            # 在齐次坐标系中，等价于右乘平移矩阵
            t_mat = translate_mat(current_joint.x, current_joint.y, current_joint.z)
            foot_position = foot_position @ t_mat
            
            if i > 1:
                # foot_position.RotateY(...)
                r_mat = rotate_y_mat(prev_joint.theta)
                foot_position = foot_position @ r_mat
                
        return foot_position

    def foot_from_base(self) -> np.ndarray:
        """正向运动学：计算足端相对于基座(Base)的 4x4 变换矩阵"""
        foot_position = np.eye(4)
        
        hip_to_foot = self.foot_from_hip()
        
        # foot_position.p = foot_from_hip().p;
        # 提取 translation vector (4x4 矩阵的第4列前3行)，赋值给当前矩阵的平移部分
        foot_position[0:3, 3] = hip_to_foot[0:3, 3]
        
        # foot_position.RotateX(...)
        foot_position = foot_position @ rotate_x_mat(self.hip.theta)
        
        # foot_position.Translate(...)
        foot_position = foot_position @ translate_mat(self.hip.x, self.hip.y, self.hip.z)
        
        return foot_position

    def zero_stance(self) -> np.ndarray:
        """计算零位站立姿态，返回一个纯平移的 4x4 矩阵"""
        if self.gait_config is None:
            raise ValueError("GaitConfig is not set")
            
        x = self.hip.x + self.upper_leg.x + self.gait_config.com_x_translation
        y = self.hip.y + self.upper_leg.y
        z = self.hip.z + self.upper_leg.z + self.lower_leg.z + self.foot.z
        
        # 将结果存为 4x4 平移矩阵
        self._zero_stance = translate_mat(x, y, z)
        return self._zero_stance

    def get_center_to_nominal(self) -> float:
        x = self.hip.x + self.upper_leg.x
        y = self.hip.y + self.upper_leg.y
        return math.sqrt(x**2 + y**2)

    # ------------------------------------------
    # 关节设置及 Getter/Setter 保持不变
    # ------------------------------------------
    def set_joints(self, hip_joint: float, upper_leg_joint: float, lower_leg_joint: float) -> None:
        self.hip.theta = hip_joint
        self.upper_leg.theta = upper_leg_joint
        self.lower_leg.theta = lower_leg_joint

    def set_joints_array(self, joints_array: List[float]) -> None:
        for i in range(3):
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