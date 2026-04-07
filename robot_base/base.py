import numpy as np
from typing import List, Optional

# 假设这些类在同级目录的其他文件中已经定义好
from robot_base.datatypes import Velocities, GaitConfig
from robot_base.leg import QuadrupedLeg

class QuadrupedBase:
    def __init__(self, gait_conf: Optional[GaitConfig] = None):
        """
        初始化四足机器人的基座类。
        在 Python 中，我们合并了无参和有参构造函数。
        """
        self.speed_: Velocities = Velocities()
        
        # 实例化四条腿：左前(lf), 右前(rf), 左后(lh), 右后(rh)
        self.lf: QuadrupedLeg = QuadrupedLeg()
        self.rf: QuadrupedLeg = QuadrupedLeg()
        self.lh: QuadrupedLeg = QuadrupedLeg()
        self.rh: QuadrupedLeg = QuadrupedLeg()

        # 将四条腿按固定顺序存入列表，方便循环遍历
        self.legs: List[QuadrupedLeg] = [
            self.lf, 
            self.rf, 
            self.lh, 
            self.rh
        ]

        # 默认初始化一个 GaitConfig
        self.gait_config: GaitConfig = GaitConfig()

        # 如果传入了外部配置，则应用配置
        if gait_conf is not None:
            self.set_gait_config(gait_conf)
        else:
            self.set_gait_config(self.gait_config)

    def _get_knee_direction(self, direction_char: str) -> int:
        """
        解析膝盖朝向字符。对应 C++ 的 getKneeDirection。
        Python 中没有 switch 语句，通常使用 if/elif 或字典映射。
        """
        if direction_char == '>':
            return -1
        elif direction_char == '<':
            return 1
        else:
            return -1

    def get_joint_positions(self) -> List[float]:
        """
        获取当前机器人的所有关节角度。
        返回包含 12 个 float 的列表 [lf_hip, lf_up, lf_low, rf_hip, ...]
        """
        joint_positions = []
        for leg in self.legs:
            joint_positions.append(leg.hip.theta)
            joint_positions.append(leg.upper_leg.theta)
            joint_positions.append(leg.lower_leg.theta)
            
        return joint_positions

    def get_foot_positions(self) -> List[np.ndarray]:
        """
        获取四条腿足端相对于基座(Base)坐标系的 4x4 变换矩阵。
        对应 C++ 中使用 geometry::Transformation 数组的逻辑。
        """
        foot_positions = []
        for leg in self.legs:
            foot_positions.append(leg.foot_from_base())
            
        return foot_positions

    def update_joint_positions(self, joints: List[float]) -> None:
        """
        更新机器人的关节角度。
        输入应该为读取到的实际电机的角度值，长度必须为 12。
        """
        if len(joints) != 12:
            raise ValueError(f"Expected 12 joint angles, got {len(joints)}")

        for i, leg in enumerate(self.legs):
            index = i * 3
            leg.hip.theta = joints[index]
            leg.upper_leg.theta = joints[index + 1]
            leg.lower_leg.theta = joints[index + 2]

    def set_gait_config(self, gait_conf: GaitConfig) -> None:
        """
        设置全局步态配置，并将其同步给每一条腿。
        """
        self.gait_config = gait_conf

        # GaitConfig 的 knee_orientation 通常是一个类似 ">>" 或 "><" 的字符串
        # ">>" 表示前腿和后腿膝盖都向后弯（狗腿结构）
        # "><" 表示前腿向后，后腿向前（蜘蛛结构）
        knee_str = self.gait_config.knee_orientation
        if len(knee_str) < 2:
            knee_str = ">>" # 防止字符串越界的保护措施

        for i, leg in enumerate(self.legs):
            leg.leg_id = i
            
            # 索引 0, 1 是前腿(lf, rf)，提取第 0 个字符
            if i < 2:
                dir_val = self._get_knee_direction(knee_str[0])
            # 索引 2, 3 是后腿(lh, rh)，提取第 1 个字符
            else:
                dir_val = self._get_knee_direction(knee_str[1])
                
            leg.is_pantograph = self.gait_config.pantograph_leg
            leg.knee_direction = dir_val

            # 将全局配置的引用传给每一条腿
            leg.gait_config = self.gait_config

    def setup_robot_geometry(self, base_x: float, base_y: float, 
                             hip_length: float, upper_length: float, lower_length: float) -> None:
        """
        初始化机器人的运动学连杆参数 (单位: 米)。
        使用 set_translation 给 Joint 类内部的 Point 对象赋值。
        
        :param base_x: 机身中心到髋关节的 X 轴距离 (纵向)
        :param base_y: 机身中心到髋关节的 Y 轴距离 (横向)
        :param hip_length: 髋关节电机到大腿电机的横向偏移 (外展/内收轴)
        :param upper_length: 大腿连杆的长度
        :param lower_length: 小腿连杆的长度
        """
        
        # ---------------------------------------------------------
        # 左前腿 (LF) - 位于第一象限 (+, +)
        # ---------------------------------------------------------
        # 1. 髋关节相对于机身中心的平移
        self.lf.hip.set_translation(base_x, base_y, 0.0)
        # 2. 大腿关节相对于髋关节的平移 (向外延展 hip_length)
        self.lf.upper_leg.set_translation(0.0, hip_length, 0.0)
        # 3. 小腿关节(膝盖)相对于大腿关节的平移 (向下延展 upper_length)
        self.lf.lower_leg.set_translation(0.0, 0.0, -upper_length)
        # 4. 足端相对于小腿关节的平移 (向下延展 lower_length)
        self.lf.foot.set_translation(0.0, 0.0, -lower_length)

        # ---------------------------------------------------------
        # 右前腿 (RF) - 位于第四象限 (+, -)
        # ---------------------------------------------------------
        self.rf.hip.set_translation(base_x, -base_y, 0.0)
        self.rf.upper_leg.set_translation(0.0, -hip_length, 0.0) # 向右外展为负
        self.rf.lower_leg.set_translation(0.0, 0.0, -upper_length)
        self.rf.foot.set_translation(0.0, 0.0, -lower_length)

        # ---------------------------------------------------------
        # 左后腿 (LH) - 位于第二象限 (-, +)
        # ---------------------------------------------------------
        self.lh.hip.set_translation(-base_x, base_y, 0.0)
        self.lh.upper_leg.set_translation(0.0, hip_length, 0.0)
        self.lh.lower_leg.set_translation(0.0, 0.0, -upper_length)
        self.lh.foot.set_translation(0.0, 0.0, -lower_length)

        # ---------------------------------------------------------
        # 右后腿 (RH) - 位于第三象限 (-, -)
        # ---------------------------------------------------------
        self.rh.hip.set_translation(-base_x, -base_y, 0.0)
        self.rh.upper_leg.set_translation(0.0, -hip_length, 0.0)
        self.rh.lower_leg.set_translation(0.0, 0.0, -upper_length)
        self.rh.foot.set_translation(0.0, 0.0, -lower_length)