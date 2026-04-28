import math
import numpy as np

# 假设 QuadrupedLeg 在同级模块中
from robot_base.leg_dh import QuadrupedLeg

class TrajectoryPlanner:
    def __init__(self, leg: QuadrupedLeg):
        self.leg_: QuadrupedLeg = leg
        self.total_control_points_: int = 12

        # 使用 numpy 单位阵来初始化上一帧的足端位置
        self.prev_foot_position_: np.ndarray = np.eye(4)

        # 预先计算好的阶乘，用于加速贝塞尔曲线的二项式系数计算
        self.factorial_ = [
            1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 
            40320.0, 362880.0, 3628800.0, 39916800.0, 479001600.0
        ]
        
        # 归一化后的贝塞尔曲线参考控制点
        self.ref_control_points_x_ = [
            -0.15, -0.2805, -0.3, -0.3, -0.3, 0.0, 
            0.0, 0.0, 0.3032, 0.3032, 0.2826, 0.15
        ]
        self.ref_control_points_y_ = [
            -0.5, -0.5, -0.3611, -0.3611, -0.3611, -0.3611, 
            -0.3611, -0.3214, -0.3214, -0.3214, -0.5, -0.5
        ]
        
        # 实际运行时使用的控制点（根据步长和高度缩放后）
        self.control_points_x_ = [0.0] * 12
        self.control_points_y_ = [0.0] * 12

        self.height_ratio_: float = 0.0
        self.length_ratio_: float = 0.0
        self.run_once_: bool = False

    def update_control_points_height(self, swing_height: float) -> None:
        """根据当前的抬腿高度参数，按比例缩放 Y 轴的控制点"""
        new_height_ratio = swing_height / 0.15
        
        if self.height_ratio_ != new_height_ratio:
            self.height_ratio_ = new_height_ratio
            for i in range(12):
                self.control_points_y_[i] = -((self.ref_control_points_y_[i] * self.height_ratio_) + (0.5 * self.height_ratio_))

    def update_control_points_length(self, step_length: float) -> None:
        """根据当前的步长参数，按比例缩放 X 轴的控制点"""
        new_length_ratio = step_length / 0.4
        
        if self.length_ratio_ != new_length_ratio:
            self.length_ratio_ = new_length_ratio
            for i in range(12):
                if i == 0:
                    self.control_points_x_[i] = -step_length / 2.0
                elif i == 11:
                    self.control_points_x_[i] = step_length / 2.0
                else:
                    self.control_points_x_[i] = self.ref_control_points_x_[i] * self.length_ratio_

    def generate(self, foot_position: np.ndarray, step_length: float, rotation: float, 
                 swing_phase_signal: float, stance_phase_signal: float) -> None:
        """
        生成轨迹，直接修改传入的 foot_position 矩阵的平移部分。
        """
        if self.leg_.gait_config is None:
            return

        self.update_control_points_height(self.leg_.gait_config.swing_height)

        # 确保首次运行时 prev_foot_position_ 有效
        if not self.run_once_:
            self.run_once_ = True
            # 使用 numpy.copy 防止对象引用污染
            self.prev_foot_position_ = np.copy(foot_position)

        # 检查是否需要迈步，如果步长为0（静止），直接返回当前位置
        if step_length == 0.0:
            self.prev_foot_position_ = np.copy(foot_position)
            self.leg_.gait_phase = True # C++ 中为 1
            return

        self.update_control_points_length(step_length)

        n = self.total_control_points_ - 1
        x = 0.0
        y = 0.0

        # ==========================================
        # 支撑相计算 (Stance Phase)
        # ==========================================
        if stance_phase_signal > swing_phase_signal:
            self.leg_.gait_phase = True
            
            x = (step_length / 2.0) * (1.0 - (2.0 * stance_phase_signal))
            y = -self.leg_.gait_config.stance_depth * math.cos((math.pi * x) / step_length)
            
        # ==========================================
        # 摆动相计算 (Swing Phase) - 贝塞尔曲线
        # ==========================================
        elif stance_phase_signal < swing_phase_signal:
            self.leg_.gait_phase = False
            
            for i in range(self.total_control_points_):
                # 计算二项式系数 (nCr)
                coeff = self.factorial_[n] / (self.factorial_[i] * self.factorial_[n - i])
                
                # 贝塞尔公式累计求和
                term = coeff * math.pow(swing_phase_signal, i) * math.pow((1.0 - swing_phase_signal), (n - i))
                x += term * self.control_points_x_[i]
                y -= term * self.control_points_y_[i] # C++ 中这里是 -=

        # ==========================================
        # 坐标变换：将 2D 轨迹投影到 3D 空间 (包含 Yaw 旋转)
        # foot_position 是 4x4 矩阵，[0,3]=X, [1,3]=Y, [2,3]=Z
        # ==========================================
        foot_position[0, 3] += x * math.cos(rotation)
        foot_position[1, 3] += x * math.sin(rotation)
        foot_position[2, 3] += y

        # 步态信号都为 0 时（可能处于过度状态），保持上一帧位置
        if swing_phase_signal == 0.0 and stance_phase_signal == 0.0 and step_length > 0.0:
            # 将上一帧的平移量赋给当前帧
            foot_position[0:3, 3] = self.prev_foot_position_[0:3, 3]

        self.prev_foot_position_ = np.copy(foot_position)