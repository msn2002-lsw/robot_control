import numpy as np
from typing import List, Optional

# 假设这些类在同级目录的其他文件中已经定义好
from .datatypes import Velocities, GaitConfig
from .leg import QuadrupedLeg

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