import math
import numpy as np
from typing import List, Tuple

# 假设以下类在同级或对应模块中已定义好
from ..robot_base.base import QuadrupedBase
from ..robot_base.datatypes import Velocities
from .trajectory_planner import TrajectoryPlanner
from .phase_generator import PhaseGenerator
from ..robot_base.leg import QuadrupedLeg

class LegController:
    def __init__(self, quadruped_base: QuadrupedBase, current_time: int = None):
        self.base_: QuadrupedBase = quadruped_base
        
        # 初始化相位生成器
        self.phase_generator = PhaseGenerator(self.base_, current_time)
        
        # 为四条腿各自实例化一个独立的轨迹规划器
        self.lf_planner = TrajectoryPlanner(self.base_.lf)
        self.rf_planner = TrajectoryPlanner(self.base_.rf)
        self.lh_planner = TrajectoryPlanner(self.base_.lh)
        self.rh_planner = TrajectoryPlanner(self.base_.rh)
        
        # 存入列表，方便后续按索引进行循环计算
        self.trajectory_planners_: List[TrajectoryPlanner] = [
            self.lf_planner,
            self.rf_planner,
            self.lh_planner,
            self.rh_planner
        ]

    @staticmethod
    def cap_velocities(velocity: float, min_velocity: float, max_velocity: float) -> float:
        """限制速度在设定的最大最小值范围内 (Clamping)"""
        return max(min_velocity, min(velocity, max_velocity))

    @staticmethod
    def transform_leg(leg: QuadrupedLeg, step_x: float, step_y: float, theta: float) -> Tuple[float, float]:
        """
        计算单条腿预期的步长和旋转角度。
        返回: (step_length, rotation)
        """
        # 获取零位站立时的坐标矩阵，并提取其 X 和 Y 坐标
        zero_stance_mat = leg.zero_stance()
        zx = zero_stance_mat[0, 3]
        zy = zero_stance_mat[1, 3]

        # 1. 施加平移 (Translate)
        tx = zx + step_x
        ty = zy + step_y

        # 2. 绕基座Z轴施加旋转 (RotateZ)
        # 这里直接使用 2D 旋转矩阵展开式，避免 4x4 矩阵相乘的性能开销
        c = math.cos(theta)
        s = math.sin(theta)
        new_x = tx * c - ty * s
        new_y = tx * s + ty * c

        # 计算从原始零位到新预测位置的增量
        delta_x = new_x - zx
        delta_y = new_y - zy

        # 因为这只是从原点到落脚点的距离（也就是半步长），所以乘以 2 得到完整步长
        step_length = math.sqrt(delta_x**2 + delta_y**2) * 2.0
        
        # 计算脚部轨迹的平面朝向角 (Yaw)
        rotation = math.atan2(delta_y, delta_x)

        return step_length, rotation

    @staticmethod
    def raibert_heuristic(stance_duration: float, target_velocity: float) -> float:
        """雷伯特启发式控制公式"""
        return (stance_duration / 2.0) * target_velocity

    def velocity_command(self, foot_positions: List[np.ndarray], req_vel: Velocities, current_time: int = None) -> None:
        """
        主控方法：输入目标速度，输出/更新四条腿的 3D 坐标矩阵
        :param foot_positions: 包含4个 4x4 numpy 矩阵的列表 (按 LF, RF, LH, RH 顺序)
        :param req_vel: 外部输入的目标线速度和角速度
        :param current_time: 当前时间戳
        """
        gait_cfg = self.base_.gait_config

        # 1. 限制输入速度，防止超过配置的硬件极限
        req_vel.linear.x = self.cap_velocities(req_vel.linear.x, -gait_cfg.max_linear_velocity_x, gait_cfg.max_linear_velocity_x)
        req_vel.linear.y = self.cap_velocities(req_vel.linear.y, -gait_cfg.max_linear_velocity_y, gait_cfg.max_linear_velocity_y)
        req_vel.angular.z = self.cap_velocities(req_vel.angular.z, -gait_cfg.max_angular_velocity_z, gait_cfg.max_angular_velocity_z)

        # 2. 计算合成速度
        # tangential_velocity = 角速度 * 半径 (从中心到前腿的名义距离)
        tangential_velocity = req_vel.angular.z * self.base_.lf.get_center_to_nominal()
        # 整体合成速度大小
        velocity_magnitude = math.sqrt(req_vel.linear.x**2 + (req_vel.linear.y + tangential_velocity)**2)

        # 3. 利用 Raibert 公式计算目标偏移量
        step_x = self.raibert_heuristic(gait_cfg.stance_duration, req_vel.linear.x)
        step_y = self.raibert_heuristic(gait_cfg.stance_duration, req_vel.linear.y)
        step_theta_arc = self.raibert_heuristic(gait_cfg.stance_duration, tangential_velocity)
        
        # 将圆弧长度转换为中心旋转角 theta (小角度近似下的弦长公式)
        theta = math.sin((step_theta_arc / 2.0) / self.base_.lf.get_center_to_nominal()) * 2.0

        step_lengths = [0.0, 0.0, 0.0, 0.0]
        trajectory_rotations = [0.0, 0.0, 0.0, 0.0]
        sum_of_steps = 0.0

        # 4. 计算每条腿的实际步长和朝向角
        for i in range(4):
            sl, rot = self.transform_leg(self.base_.legs[i], step_x, step_y, theta)
            step_lengths[i] = sl
            trajectory_rotations[i] = rot
            sum_of_steps += sl

        # 5. 驱动相位生成器（节拍器更新）
        # 将4条腿的平均步长传给节拍器
        avg_step_length = sum_of_steps / 4.0
        self.phase_generator.run(velocity_magnitude, avg_step_length, current_time)

        # 6. 生成贝塞尔轨迹并更新 3D 坐标
        for i in range(4):
            # 获取节拍器输出的归一化相位信号 (0.0 -> 1.0)
            swing_signal = self.phase_generator.swing_phase_signal[i]
            stance_signal = self.phase_generator.stance_phase_signal[i]
            
            # 生成该瞬间的具体落脚点矩阵
            self.trajectory_planners_[i].generate(
                foot_positions[i], 
                step_lengths[i], 
                trajectory_rotations[i], 
                swing_signal, 
                stance_signal
            )